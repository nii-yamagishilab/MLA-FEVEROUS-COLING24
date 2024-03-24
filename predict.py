# Copyright (c) 2022, Yamagishi Laboratory, National Institute of Informatics
# Author: Shirin Dabbaghi(sdabbag@gwdg.de)
# All rights reserved.
import argparse
import numpy as np
import pytorch_lightning as pl
from datetime import datetime
from pytorch_lightning.utilities import rank_zero_info
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from train import FactCheckerTransformer


def build_args():
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--checkpoint_file", type=str, required=True)
    parser.add_argument("--in_file", type=str, required=True)
    parser.add_argument("--out_file", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()
    return args


def main():
    t_start = datetime.now()
    args = build_args()
    print("checkpoint_file", args.checkpoint_file)

    model = FactCheckerTransformer.load_from_checkpoint(
        checkpoint_path=args.checkpoint_file
    )
    model.hparams.out_file = args.out_file
    model.freeze()

    params = {}
    params["precision"] = model.hparams.precision

    trainer = pl.Trainer.from_argparse_args(args, logger=False, **params)

    test_file_path = Path(args.in_file)
    if not test_file_path.exists():
        raise RuntimeError(f"Cannot find '{test_file_path}'")
    feature_list = model.create_features("test", test_file_path)
    test_dataloader = DataLoader(
        TensorDataset(*feature_list),
        batch_size=args.batch_size,
        shuffle=False,
    )

    predictions = trainer.predict(model, test_dataloader)
    probs = np.vstack([p for p in predictions])
    rank_zero_info(f"Save output probabilities to {args.out_file}")
    np.savetxt(args.out_file, probs, delimiter=" ", fmt="%.5f")

    t_delta = datetime.now() - t_start
    rank_zero_info(f"Prediction took '{t_delta}'")


if __name__ == "__main__":
    main()
