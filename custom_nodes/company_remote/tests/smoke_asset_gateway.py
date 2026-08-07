from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import server
import torch
from PIL import Image


class _Routes:
    def __getattr__(self, _name):
        def route(*_args, **_kwargs):
            return lambda function: function

        return route


if not hasattr(server.PromptServer, "instance"):
    server.PromptServer.instance = type("PromptServerInstance", (), {"routes": _Routes()})()

from custom_nodes.company_remote.asset_gateway import create_seedance_image_asset


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test TOS upload and /v1/assets registration.")
    parser.add_argument("image", nargs="?", default="input/example.png")
    parser.add_argument("--reuse-cached", action="store_true")
    args = parser.parse_args()

    path = Path(args.image)
    with Image.open(path) as source:
        array = np.asarray(source.convert("RGB"), dtype=np.float32) / 255.0
    asset_id, report_text, cache_reused = create_seedance_image_asset(
        torch.from_numpy(array)[None,],
        character_label="real-smoke",
        reuse_cached=args.reuse_cached,
    )
    report = json.loads(report_text)
    print(
        json.dumps(
            {
                "asset_id": asset_id,
                "width": report["image"]["width"],
                "height": report["image"]["height"],
                "cache_reused": cache_reused,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
