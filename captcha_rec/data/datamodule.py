from __future__ import annotations

import glob
from dataclasses import dataclass
from os.path import join
from pathlib import Path
from typing import Optional, Tuple

import cv2
import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset

from captcha_rec.data.preprocessing import build_transforms


@dataclass(frozen=True)
class Vocabulary:
    tokens: Tuple[str, ...]
    token_to_id: dict[str, int]
    id_to_token: dict[int, str]
    pad_token: str = "<pad>"

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.pad_token]

    def size(self) -> int:
        return len(self.tokens)


def build_default_vocab() -> Vocabulary:
    special = ("<pad>",)
    digits = tuple(str(i) for i in range(10))
    upper = tuple(chr(c) for c in range(ord("A"), ord("Z") + 1))
    lower = tuple(chr(c) for c in range(ord("a"), ord("z") + 1))
    tokens = special + digits + upper + lower
    token_to_id = {t: i for i, t in enumerate(tokens)}
    id_to_token = {i: t for t, i in token_to_id.items()}
    return Vocabulary(
        tokens=tokens,
        token_to_id=token_to_id,
        id_to_token=id_to_token,
    )


class ImageToTextDataset(Dataset):
    def __init__(self, path, transform):
        self.vocab = build_default_vocab()
        BAN_DATA = [
            f"{path}/Large_Captcha_Dataset/4q2wA.png",
        ]
        self.MAX_LEN = 10

        self.path = path
        self.transformer = transform
        self.file = []

        file_list = glob.glob(join(self.path, "*"))
        self.file = [
            file
            for file in file_list
            if (file.endswith(".png") or file.endswith(".jpg"))
        ]
        for ban_file in BAN_DATA:
            if ban_file in self.file:
                self.file.remove(ban_file)
        self.num = len(self.file)

    def __len__(self):
        return self.num

    def transform(self, image):
        if self.transformer is not None:
            return self.transformer(image)
        else:
            return image

    def __getitem__(self, idx):
        filename = self.file[idx]

        Y = []
        for char in list(filename.split("/")[-1].split(".")[0]):
            Y.append(self.vocab.token_to_id[char])

        if len(Y) < self.MAX_LEN:
            Y += [self.vocab.token_to_id["<pad>"]] * (self.MAX_LEN - len(Y))

        img = cv2.imread(self.file[idx])
        try:
            sketch_image = cv2.cvtColor(img[:, :256, :], cv2.COLOR_BGR2RGB)
        except Exception:
            print(self.file[idx])
        X = self.transform(sketch_image)

        Y_tensor_list = []
        for y_ind in Y:
            y_tensor = torch.zeros(self.vocab.size())
            y_tensor[y_ind] = 1
            Y_tensor_list.append(y_tensor.unsqueeze(0))

        return X, torch.tensor(Y)


class CaptchaDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_root: str,
        image_size: int,
        batch_size: int,
        num_workers: int,
        max_len: int,
        pin_memory: bool = True,
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.image_size = int(image_size)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.max_len = int(max_len)
        self.pin_memory = bool(pin_memory)

        self.vocab = build_default_vocab()
        self.train_ds: Optional[ImageToTextDataset] = None
        self.val_ds: Optional[ImageToTextDataset] = None

        self.tfms = build_transforms(self.image_size)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab.tokens)

    @property
    def pad_id(self) -> int:
        return self.vocab.pad_id

    def prepare_data(self) -> None:
        # Do not download here; controlled by DVC/download_data() in commands.
        # Keeping Lightning best practice: no side effects in prepare_data.
        return

    def setup(self, stage=None) -> None:
        dataset = ImageToTextDataset(self.data_root, self.tfms)
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [len(dataset) - len(dataset) // 10, len(dataset) // 10]
        )
        self.train_ds = train_dataset
        self.val_ds = test_dataset

    def train_dataloader(self) -> DataLoader:
        if self.train_ds is None:
            raise RuntimeError("Call setup() before train_dataloader().")
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_ds is None:
            raise RuntimeError("Call setup() before val_dataloader().")
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def test_dataloader(self) -> DataLoader:
        if self.val_ds is None:
            raise RuntimeError("Call setup() before test_dataloader().")
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )
