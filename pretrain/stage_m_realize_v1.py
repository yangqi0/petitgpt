#!/usr/bin/env python3
"""Stage-M realization v1: candidate plan, authorization, native packing, strict verification.

The Stage-I-native Stage-M entrypoint. It consumes the *accepted* Stage-I publication directly
and packs it into two canonical schema-3 releases, one per stage stream.

What this path deliberately does not contain, and cannot reach: quota allocation, source
re-selection, weighted interleave, source scheduling, validation reservation, ``val_ratio`` or
``min_val_tokens_per_source``. Those belong to the legacy selector-v1 orchestration in
``build_pretrain_shards.write_shards``, which this module never calls. The only primitives
borrowed from that file are the canonical tokenizer loader and ``encode_with_accounting`` --
the shared encoding core whose agreement with Stage-I accounting is a measured fact, not an
assumption (DECISIONS D-129).

Authority order, each layer only ever reading upward:

1. the externally authorized candidate plan, checked against owner-supplied bytes;
2. the accepted Stage-I publication and every plan-bound resource, re-derived from disk;
3. the published Stage-M releases, which must prove themselves equal to the plan's expected
   accounting and may never define it.

The plan is revalidated from disk twice: once before a single output byte exists, and again
after packing and before the canonical completion object is written. A late edit to the plan,
to an implementation file, or to the accepted Stage-I bytes therefore still costs nothing to
reject.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pretrain.build_pretrain_shards import encode_with_accounting, load_tokenizer  # noqa: E402
from pretrain.stage_m_contract_v1 import (  # noqa: E402
    CANDIDATE_PLAN_SCHEMA,
    MODEL_CONTRACT,
    ORDERING_CONTRACT_ID,
    PACKING_SEMANTICS,
    RELEASE_PROFILE,
    SEQ_LEN,
    STAGE_STREAMS,
    Environment,
    StageMError,
    StreamAccounting,
    canonical_json_bytes,
    current_environment,
    file_sha256,
    m_implementation_bundle,
    require,
    require_int,
    resolve_repo_root,
    sha256_hex,
    stream_accounting,
    total_accounting,
    validated_sha256,
    verify_environment,
)
from pretrain.stage_m_input_v1 import (  # noqa: E402
    AcceptedStageI,
    commitments_as_canonical,
    derive_input_sequence_commitments,
    iter_accepted_records,
    load_accepted_stage_i,
)
from pretrain.stage_m_output_v1 import (  # noqa: E402
    DEFAULT_SHARD_TOKENS,
    build_release_meta,
    discard_staging,
    pack_stream,
    publish_release_atomic,
    staging_directory,
    verify_release_against_accounting,
    write_manifest,
)
from src.special_tokens import BOS_ID, EOS_ID, assert_tokenizer_contract  # noqa: E402

AUTHORIZATION_STATUS_UNAUTHORIZED = "NOT_AUTHORIZED"
PLAN_AUTHORIZATION_NOTE = (
    "This plan carries no owner authorization and cannot create one. Stage-M production must be "
    "invoked with --expected-plan-sha256 supplied externally by the owner, and that "
    "authorization applies only to these exact plan bytes, this accepted Stage-I publication, "
    "these resources and this implementation bundle."
)


def _git_head(repo_root: Path) -> str:
    """Provenance metadata only. Byte verification never depends on this value."""
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.stdout.strip() or "unknown"


# --------------------------------------------------------------------- resources


def _bind_resource(repo_root: Path, relative: str, *, label: str) -> dict[str, Any]:
    path = (repo_root / relative).resolve()
    require(path.is_file(), f"{label} is missing: {path}")
    return {
        "path": relative,
        "sha256": file_sha256(path),
        "size_bytes": int(path.stat().st_size),
    }


def _verify_resource(repo_root: Path, bound: Mapping[str, Any], *, label: str) -> Path:
    relative = str(bound["path"])
    path = (repo_root / relative).resolve()
    require(path.is_file(), f"{label} is missing: {path}")
    actual = file_sha256(path)
    require(
        actual == bound["sha256"],
        f"{label} bytes changed: actual={actual}, plan={bound['sha256']}",
    )
    require(
        int(path.stat().st_size) == int(bound["size_bytes"]),
        f"{label} size changed: {path}",
    )
    return path


def _exclusion_binding(repo_root: Path, relative: str) -> dict[str, Any]:
    """Bind the reference-exclusion manifest Stage I enforced, and read its real hash count.

    Stage M does not re-filter: the exclusion was applied at Stage I, and this binding records
    which manifest governed the corpus so the release can declare it truthfully and Stage P can
    prove the Stage-A/Stage-B/reference releases share one exclusion identity.
    """
    bound = _bind_resource(repo_root, relative, label="reference exclusion manifest")
    with open((repo_root / relative).resolve(), encoding="utf-8") as handle:
        payload = json.load(handle)
    require(isinstance(payload, dict), "reference exclusion manifest must be a JSON object")
    hash_count = require_int(payload.get("hash_count"), field="exclusion.hash_count", minimum=1)
    bound["hash_count"] = hash_count
    return bound


def release_exclusion_block(bound: Mapping[str, Any]) -> dict[str, Any]:
    """The schema-3 ``reference_validation_exclusion`` object for a Stage-M release."""
    return {
        "enabled": True,
        "manifest_count": 1,
        "union_hash_count": int(bound["hash_count"]),
        "enforced_at_stage": "stage_i",
        "reapplied_by_stage_m": False,
        "manifests": [
            {
                "enabled": True,
                "path": str(bound["path"]),
                "manifest_sha256": str(bound["sha256"]),
                "hash_count": int(bound["hash_count"]),
            }
        ],
    }


# --------------------------------------------------------------------- candidate plan


def generate_candidate_plan(
    *,
    repo_root: Path,
    accepted: AcceptedStageI,
    commitments: Mapping[str, Any],
    tokenizer_relative: str,
    exclusion_relative: str,
    environment: Environment,
    shard_tokens: int,
    implementation_commit: str | None = None,
) -> dict[str, Any]:
    """Build the unauthorized candidate Stage-M plan.

    Everything load-bearing is derived here from bytes on disk: the accepted publication, the
    resources, the implementation bundle and the environment. Nothing is copied from a caller
    argument that a later run could substitute.
    """
    tokenizer = _bind_resource(repo_root, tokenizer_relative, label="canonical tokenizer")
    exclusion = _exclusion_binding(repo_root, exclusion_relative)
    bundle_files, bundle_digest = m_implementation_bundle(repo_root)

    streams: dict[str, Any] = {}
    accountings: list[StreamAccounting] = []
    for stage in STAGE_STREAMS:
        declared = accepted.stage_membership[stage]
        commitment = commitments[stage]
        require(
            int(commitment["serialized_tokens"]) == int(declared["serialized_tokens"]),
            f"{stage}: commitment totals disagree with the accepted publication",
        )
        accounting = stream_accounting(stage, int(declared["serialized_tokens"]), SEQ_LEN)
        accountings.append(accounting)
        streams[stage] = {
            "input_record_count": int(commitment["record_count"]),
            "input_serialized_tokens": int(commitment["serialized_tokens"]),
            "input_content_tokens": int(commitment["content_tokens"]),
            "input_sequence_commitment": str(commitment["commitment"]),
            "input_sequence_commitment_schema": str(commitment["schema_version"]),
            "expected_accounting": accounting.as_canonical(),
        }

    return {
        "schema_version": CANDIDATE_PLAN_SCHEMA,
        "authorization_status": AUTHORIZATION_STATUS_UNAUTHORIZED,
        "authorization_note": PLAN_AUTHORIZATION_NOTE,
        "implementation_commit": implementation_commit or _git_head(repo_root),
        "implementation_bundle_sha256": bundle_digest,
        "implementation_files": dict(sorted(bundle_files.items())),
        "environment_contract": environment.as_canonical(),
        "accepted_stage_i": accepted.as_canonical(),
        "accepted_stage_i_identity_sha256": accepted.identity_sha256(),
        "resources": {
            "tokenizer": tokenizer,
            "reference_exclusion_manifest": exclusion,
        },
        "model_contract": dict(MODEL_CONTRACT),
        "ordering_contract": {
            "policy": ORDERING_CONTRACT_ID,
            "rule": (
                "per stage, the accepted Stage-I publication filtered to that stage with its "
                "accepted physical relative order preserved exactly"
            ),
            "weighted_interleave": False,
            "selection_rank_reorder": False,
            "source_quota_scheduling": False,
            "hash_shuffle": False,
            "new_random_permutation": False,
            "only_shuffle": "training-time block-ID permutation",
        },
        "packing_semantics": dict(PACKING_SEMANTICS),
        "release_profile": {**dict(RELEASE_PROFILE), "shard_tokens": int(shard_tokens)},
        "stage_streams": streams,
        "expected_totals": total_accounting(accountings),
        "text_field": "training_text",
        "legacy_orchestration_used": False,
    }


# --------------------------------------------------------------------- authorization


@dataclass(frozen=True)
class AuthorizedMContext:
    """A Stage-M authorization. Only :func:`authorize_plan` can produce one."""

    repo_root: Path
    plan_path: Path
    plan_sha256: str
    plan: Mapping[str, Any]
    accepted: AcceptedStageI
    tokenizer_path: Path
    exclusion: Mapping[str, Any]
    environment: Environment
    bundle_sha256: str
    state_sha256: str

    def stage_accounting(self, stage: str) -> StreamAccounting:
        declared = self.plan["stage_streams"][stage]["expected_accounting"]
        return stream_accounting(
            stage, int(declared["input_serialized_tokens"]), int(declared["seq_len"])
        )


def _derive_state(
    plan_path: Path, expected_plan_sha256: str, repo_root: Path | None
) -> AuthorizedMContext:
    root = resolve_repo_root(repo_root)
    path = Path(plan_path).expanduser().resolve()
    require(path.is_file(), f"candidate Stage-M plan is missing: {path}")
    payload = path.read_bytes()
    actual = sha256_hex(payload)
    expected = validated_sha256(expected_plan_sha256, field="--expected-plan-sha256")
    require(
        actual == expected,
        f"candidate Stage-M plan digest mismatch: actual={actual}, authorized={expected}",
    )
    plan = json.loads(payload.decode("utf-8"))
    require(isinstance(plan, dict), "candidate Stage-M plan must be a JSON object")
    require(
        plan.get("schema_version") == CANDIDATE_PLAN_SCHEMA,
        f"unsupported Stage-M plan schema: {plan.get('schema_version')!r}",
    )
    require(
        plan.get("authorization_status") == AUTHORIZATION_STATUS_UNAUTHORIZED,
        "a Stage-M plan must not carry a self-declared authorization; owner authorization is "
        "the externally supplied digest only",
    )
    require(plan.get("legacy_orchestration_used") is False, "plan declares legacy orchestration")
    require(plan.get("text_field") == "training_text", "plan does not bind training_text")

    environment = current_environment()
    verify_environment(environment)
    require(
        environment.as_canonical() == plan.get("environment_contract"),
        f"runtime environment differs from the authorized plan: {environment.as_canonical()} "
        f"!= {plan.get('environment_contract')}",
    )

    files, bundle_digest = m_implementation_bundle(root)
    require(
        bundle_digest == plan.get("implementation_bundle_sha256"),
        f"Stage-M implementation bundle changed: actual={bundle_digest}, "
        f"plan={plan.get('implementation_bundle_sha256')}",
    )
    require(
        files == dict(plan.get("implementation_files") or {}),
        "Stage-M implementation member digests differ from the authorized plan",
    )

    tokenizer = _verify_resource(root, plan["resources"]["tokenizer"], label="canonical tokenizer")
    assert_tokenizer_contract(str(tokenizer))
    exclusion = plan["resources"]["reference_exclusion_manifest"]
    _verify_resource(root, exclusion, label="reference exclusion manifest")

    require(
        dict(plan.get("model_contract") or {}) == dict(MODEL_CONTRACT),
        "plan model contract differs from the frozen final-model contract",
    )
    require(
        dict(plan.get("packing_semantics") or {}) == dict(PACKING_SEMANTICS),
        "plan packing semantics differ from the frozen contract",
    )
    require(
        (plan.get("ordering_contract") or {}).get("policy") == ORDERING_CONTRACT_ID,
        "plan ordering contract is not the frozen Stage-M ordering policy",
    )

    bound = plan["accepted_stage_i"]
    accepted = load_accepted_stage_i(
        root / str(bound["run_dir"]),
        expected_run_identity=str(bound["run_identity"]),
        expected_manifest_sha256=str(bound["manifest_sha256"]),
        expected_completion_sha256=str(bound["completion_object_sha256"]),
        expected_layer2_sha256=str(bound["layer2_expected_result_sha256"]),
        expected_records=int(bound["total_records"]),
        expected_serialized_tokens=int(bound["total_serialized_tokens"]),
        expected_shard_count=int(bound["shard_count"]),
    )
    require(
        accepted.as_canonical() == dict(bound),
        "the accepted Stage-I publication on disk differs from the one the plan authorized",
    )
    require(
        accepted.identity_sha256() == plan.get("accepted_stage_i_identity_sha256"),
        "accepted Stage-I identity digest differs from the authorized plan",
    )

    for stage in STAGE_STREAMS:
        declared = plan["stage_streams"][stage]
        membership = accepted.stage_membership[stage]
        require(
            int(declared["input_record_count"]) == int(membership["records"]),
            f"{stage}: plan record count differs from the accepted publication",
        )
        require(
            int(declared["input_serialized_tokens"]) == int(membership["serialized_tokens"]),
            f"{stage}: plan serialized tokens differ from the accepted publication",
        )
        expected = stream_accounting(
            stage, int(membership["serialized_tokens"]), SEQ_LEN
        ).as_canonical()
        require(
            dict(declared["expected_accounting"]) == expected,
            f"{stage}: plan expected accounting is not the frozen derivation",
        )

    state = {
        "plan_sha256": actual,
        "bundle_sha256": bundle_digest,
        "environment": environment.as_canonical(),
        "accepted_stage_i_identity_sha256": accepted.identity_sha256(),
        "tokenizer_sha256": plan["resources"]["tokenizer"]["sha256"],
        "exclusion_sha256": exclusion["sha256"],
    }
    return AuthorizedMContext(
        repo_root=root,
        plan_path=path,
        plan_sha256=actual,
        plan=plan,
        accepted=accepted,
        tokenizer_path=tokenizer,
        exclusion=exclusion,
        environment=environment,
        bundle_sha256=bundle_digest,
        state_sha256=sha256_hex(canonical_json_bytes(state)),
    )


def authorize_plan(
    plan_path: Path, expected_plan_sha256: str, repo_root: Path | None = None
) -> AuthorizedMContext:
    """Authorization is a capability, not a comparison.

    The owner supplies the expected digest out of band; it is checked against the plan bytes,
    and those same bytes then drive every load. A plan cannot authorise itself and an
    authorization for one plan cannot be reused for another.
    """
    return _derive_state(plan_path, expected_plan_sha256, repo_root)


def revalidate(context: AuthorizedMContext) -> AuthorizedMContext:
    """Re-derive the whole authorized state from disk and prove it did not move."""
    fresh = _derive_state(context.plan_path, context.plan_sha256, context.repo_root)
    require(
        fresh.state_sha256 == context.state_sha256,
        "the authorized Stage-M state changed between validations",
    )
    return fresh


# --------------------------------------------------------------------- realization


def _framed_documents(
    context: AuthorizedMContext, stage: str, tokenizer: Any
) -> Iterator[list[int]]:
    """Yield ``[BOS] content [EOS]`` id lists for one stage, in accepted physical order.

    Each accepted record is visited once, in the accepted order, and only the requested stage
    is emitted. The per-record token count is proved against the count Stage I recorded, so a
    tokenizer or text substitution cannot pass silently.
    """
    for record_stage, record in iter_accepted_records(context.accepted):
        if record_stage != stage:
            continue
        ids, content, boundary = encode_with_accounting(
            tokenizer,
            record["training_text"],
            add_bos=True,
            add_eos=True,
            bos_id=BOS_ID,
            eos_id=EOS_ID,
        )
        if (
            boundary != 2
            or len(ids) != int(record["serialized_token_count"])
            or content != int(record["content_token_count"])
        ):
            raise StageMError(
                f"{stage}: re-tokenization disagrees with the accepted Stage-I accounting for "
                f"{record['cleaned_text_sha256']}: serialized {len(ids)} vs "
                f"{record['serialized_token_count']}, content {content} vs "
                f"{record['content_token_count']}"
            )
        yield ids


def realize_stage(
    context: AuthorizedMContext,
    stage: str,
    *,
    destination: Path,
    tokenizer: Any,
) -> dict[str, Any]:
    """Pack one stage stream, validate it, and publish it atomically."""
    accounting = context.stage_accounting(stage)
    shard_tokens = int(context.plan["release_profile"]["shard_tokens"])
    staging = staging_directory(destination)
    try:
        packed = pack_stream(
            stage=stage,
            documents=_framed_documents(context, stage, tokenizer),
            accounting=accounting,
            directory=staging / "train",
            shard_tokens=shard_tokens,
        )
        require(
            packed.documents == int(context.plan["stage_streams"][stage]["input_record_count"]),
            f"{stage}: consumed {packed.documents} records, plan expects "
            f"{context.plan['stage_streams'][stage]['input_record_count']}",
        )
        meta = build_release_meta(
            packed,
            tokenizer_path=str(context.tokenizer_path),
            tokenizer_sha256=str(context.plan["resources"]["tokenizer"]["sha256"]),
            stage_m_binding=stage_m_release_binding(context, stage),
            reference_exclusion=release_exclusion_block(context.exclusion),
        )
        # Second fresh revalidation: everything is built, nothing is published yet, so a late
        # change to the plan, the implementation or the accepted input still costs nothing.
        revalidate(context)
        manifest_sha256 = write_manifest(staging, meta)
        result = verify_release_against_accounting(staging, accounting)
        published = publish_release_atomic(staging, destination)
    except BaseException:
        discard_staging(staging)
        raise
    verify_release_against_accounting(published, accounting)
    return {
        "stage": stage,
        "release_dir": str(published),
        "manifest_sha256": manifest_sha256,
        "shards": int(result["expected_shards"]),
        "stored_token_ids": int(result["expected_tokens"]),
        "accounting": accounting.as_canonical(),
    }


def stage_m_release_binding(context: AuthorizedMContext, stage: str) -> dict[str, Any]:
    """The Stage-M provenance object embedded in a published release manifest."""
    declared = context.plan["stage_streams"][stage]
    return {
        "candidate_plan_schema": CANDIDATE_PLAN_SCHEMA,
        "candidate_plan_sha256": context.plan_sha256,
        "implementation_bundle_sha256": context.bundle_sha256,
        "implementation_commit": str(context.plan.get("implementation_commit")),
        "environment": context.environment.as_canonical(),
        "ordering_policy": ORDERING_CONTRACT_ID,
        "stage": stage,
        "stage_stream_count": 2,
        "accepted_stage_i": {
            "run_identity": context.accepted.run_identity,
            "layer2_expected_result_sha256": context.accepted.layer2_sha256,
            "manifest_sha256": context.accepted.manifest_sha256,
            "completion_object_sha256": context.accepted.completion_sha256,
            "shard_count": context.accepted.shard_count,
            "total_records": context.accepted.total_records,
            "total_serialized_tokens": context.accepted.total_serialized_tokens,
            "identity_sha256": context.accepted.identity_sha256(),
        },
        "input_record_count": int(declared["input_record_count"]),
        "input_serialized_tokens": int(declared["input_serialized_tokens"]),
        "input_sequence_commitment": str(declared["input_sequence_commitment"]),
        "input_sequence_commitment_schema": str(declared["input_sequence_commitment_schema"]),
        "model_contract": dict(MODEL_CONTRACT),
    }


def realize_and_publish(context: AuthorizedMContext, *, out_dir: Path) -> dict[str, Any]:
    """Pack, validate and publish both stage streams."""
    # First fresh revalidation: before a single output byte exists.
    revalidate(context)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(str(context.tokenizer_path))
    results = {}
    for stage in STAGE_STREAMS:
        results[stage] = realize_stage(
            context, stage, destination=out_dir / stage, tokenizer=tokenizer
        )
    return {
        "out_dir": str(out_dir),
        "candidate_plan_sha256": context.plan_sha256,
        "authorized_state_sha256": context.state_sha256,
        "stages": results,
    }


# --------------------------------------------------------------------- CLI


def _write_new_file(path: Path, payload: bytes) -> None:
    require(not path.exists(), f"refusing to overwrite an existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    plan_parser = sub.add_parser("plan", help="generate an unauthorized candidate Stage-M plan")
    plan_parser.add_argument("--accepted-stage-i-dir", type=Path, required=True)
    plan_parser.add_argument("--tokenizer", type=str, required=True)
    plan_parser.add_argument("--reference-exclusion-manifest", type=str, required=True)
    plan_parser.add_argument("--out", type=Path, required=True)
    plan_parser.add_argument("--repo-root", type=Path, default=None)
    plan_parser.add_argument("--shard-tokens", type=int, default=DEFAULT_SHARD_TOKENS)
    plan_parser.add_argument("--implementation-commit", type=str, default=None)
    plan_parser.add_argument("--commitments-out", type=Path, default=None)

    sub.add_parser("verify-environment", help="check the frozen interpreter contract")

    run_parser = sub.add_parser("run", help="run the authorized Stage-M packing and publish it")
    run_parser.add_argument("--plan", type=Path, required=True)
    run_parser.add_argument("--expected-plan-sha256", type=str, required=True)
    run_parser.add_argument("--out-dir", type=Path, required=True)
    run_parser.add_argument("--repo-root", type=Path, default=None)

    verify_parser = sub.add_parser("verify", help="strictly verify a published Stage-M release")
    verify_parser.add_argument("--release", type=Path, required=True)
    verify_parser.add_argument("--plan", type=Path, required=True)
    verify_parser.add_argument("--expected-plan-sha256", type=str, required=True)
    verify_parser.add_argument("--stage", type=str, required=True, choices=list(STAGE_STREAMS))
    verify_parser.add_argument("--repo-root", type=Path, default=None)

    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))

    if args.command == "verify-environment":
        environment = current_environment()
        verify_environment(environment)
        print(canonical_json_bytes(environment.as_canonical()).decode("utf-8"), end="")
        return 0

    if args.command == "plan":
        root = resolve_repo_root(args.repo_root)
        accepted = load_accepted_stage_i(args.accepted_stage_i_dir)
        commitments = derive_input_sequence_commitments(accepted)
        canonical = commitments_as_canonical(commitments)
        if args.commitments_out is not None:
            _write_new_file(
                args.commitments_out,
                canonical_json_bytes({
                    "accepted_stage_i_run_identity": accepted.run_identity,
                    "commitments": canonical,
                }),
            )
        plan = generate_candidate_plan(
            repo_root=root,
            accepted=accepted,
            commitments=canonical,
            tokenizer_relative=args.tokenizer,
            exclusion_relative=args.reference_exclusion_manifest,
            environment=current_environment(),
            shard_tokens=int(args.shard_tokens),
            implementation_commit=args.implementation_commit,
        )
        payload = canonical_json_bytes(plan)
        _write_new_file(args.out, payload)
        print(f"{sha256_hex(payload)}  {args.out}")
        return 0

    if args.command == "verify":
        context = authorize_plan(args.plan, args.expected_plan_sha256, args.repo_root)
        accounting = context.stage_accounting(args.stage)
        result = verify_release_against_accounting(args.release, accounting)
        print(
            canonical_json_bytes({
                "verified": str(args.release),
                "stage": args.stage,
                "shards": int(result["expected_shards"]),
                "stored_token_ids": int(result["expected_tokens"]),
                "training_sequences": accounting.training_sequences,
            }).decode("utf-8"),
            end="",
        )
        return 0

    context = authorize_plan(args.plan, args.expected_plan_sha256, args.repo_root)
    summary = realize_and_publish(context, out_dir=args.out_dir)
    for stage in STAGE_STREAMS:
        entry = summary["stages"][stage]
        print(f"published {stage} {entry['release_dir']} manifest {entry['manifest_sha256']}")
    return 0


__all__ = [
    "AUTHORIZATION_STATUS_UNAUTHORIZED",
    "AuthorizedMContext",
    "authorize_plan",
    "generate_candidate_plan",
    "main",
    "realize_and_publish",
    "realize_stage",
    "release_exclusion_block",
    "revalidate",
    "stage_m_release_binding",
]


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (StageMError, RuntimeError) as exc:
        print(f"FAIL-CLOSED: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
