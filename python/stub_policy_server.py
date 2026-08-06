"""A stand-in openpi policy server that returns constant actions.

Speaks the same websocket + msgpack protocol as openpi's WebsocketPolicyServer
(metadata frame on connect, then one packed response per packed observation),
but runs no model. Lets the whole evaluation client -- sharding, rollout loop,
episodes.jsonl, resume, video writing, aggregation -- be exercised on a machine
with no GPU before spending H20 time on it.

The success rate it produces is meaningless by construction. Never aggregate a
stub run into the ledger.

    python python/stub_policy_server.py --port 8000
"""

from __future__ import annotations

import argparse
import asyncio
import logging

import numpy as np
import websockets.asyncio.server
from openpi_client import msgpack_numpy


async def serve(port: int, action_horizon: int, action_dim: int, noise: float, seed: int) -> None:
    packer = msgpack_numpy.Packer()
    rng = np.random.default_rng(seed)
    served = {"connections": 0, "infers": 0}

    async def handler(websocket):
        served["connections"] += 1
        await websocket.send(packer.pack({"stub": True, "action_horizon": action_horizon}))
        async for message in websocket:
            msgpack_numpy.unpackb(message)  # decode to prove the client's payload is well formed
            served["infers"] += 1
            actions = rng.normal(0.0, noise, size=(action_horizon, action_dim)).astype(np.float32)
            actions[:, -1] = -1.0  # keep the gripper open
            await websocket.send(packer.pack({"actions": actions, "policy_timing": {"infer_ms": 0.0}}))

    async with websockets.asyncio.server.serve(handler, "0.0.0.0", port, compression=None, max_size=None):
        logging.info("stub policy server on :%d (horizon=%d dim=%d)", port, action_horizon, action_dim)
        await asyncio.Future()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--action-horizon", type=int, default=10, help="pi05_libero predicts 10")
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--noise", type=float, default=0.05, help="stddev of the random actions")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        asyncio.run(serve(args.port, args.action_horizon, args.action_dim, args.noise, args.seed))
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
