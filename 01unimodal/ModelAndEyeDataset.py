import os
import json
import pandas as pd
from PIL import Image

from torch.utils.data import Dataset

import timm
import torch
import torch.nn as nn


class SiameseSeResNeXtdropout(nn.Module):
    def __init__(self, dropout_p=0.3, spatial_dropout_p=0.3,out=2):
        super(SiameseSeResNeXtdropout, self).__init__()

        model_name = 'seresnext50_32x4d'
        base_model = timm.create_model(model_name, pretrained=True)

        # 截取基模型的各部分
        self.initial_layers = nn.Sequential(*list(base_model.children())[:3])
        self.blocks = list(base_model.children())[3:-2]
        self.avgpool = list(base_model.children())[-2]

        # 应用Spatial Dropout到初始层和所有中间层
        self.spatial_dropout_initial = nn.Dropout2d(p=spatial_dropout_p)

        # 将blocks封装到包含Spatial Dropout的Sequential中
        enhanced_blocks = []
        for block in self.blocks:
            # 每个block后都加入Spatial Dropout
            enhanced_blocks.append(nn.Sequential(
                block,
                nn.Dropout2d(p=spatial_dropout_p)
            ))

        self.enhanced_blocks = nn.Sequential(*enhanced_blocks)

        self.dropout = nn.Dropout(p=dropout_p)
        num_features = base_model.fc.in_features
        self.fc = nn.Linear(num_features * 2, out)


    def forward(self, combined_image):
        left_eye = combined_image[:, :3, :, :]
        right_eye = combined_image[:, 3:6, :, :]

        # 处理左右眼图像
        left_features = self.spatial_dropout_initial(self.initial_layers(left_eye))
        right_features = self.spatial_dropout_initial(self.initial_layers(right_eye))

        left_features = self.enhanced_blocks(left_features)
        right_features = self.enhanced_blocks(right_features)

        left_features = self.avgpool(left_features)
        right_features = self.avgpool(right_features)

        left_features = torch.flatten(left_features, 1)
        right_features = torch.flatten(right_features, 1)

        combined_features = torch.cat([left_features, right_features], dim=1)
        combined_features = self.dropout(combined_features)

        output = self.fc(combined_features)

        return output

class NewEyeDataset(Dataset):
    def __init__(self, df, root_dir, ids, transform=None):
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        self.ids = ids

    @classmethod
    def from_json(cls, json_file, root_dir, group_value, include_0_9mm, transform=None):

        with open(str(json_file), 'r') as f:
            data = json.load(f)
        df = pd.DataFrame.from_dict(data, orient='index')
        filtered_df = df[df['group'] == group_value]
        if not include_0_9mm:
            filtered_df = filtered_df[filtered_df['thickness'] != 0.9]
        ids = filtered_df.index.tolist()
        return cls(filtered_df, root_dir, ids, transform)

    def __len__(self):

        return len(self.ids)

    def __getitem__(self, idx):
        id = self.ids[idx]
        pair = self.df.loc[id]
        left_eye_path = os.path.join(self.root_dir, pair["left_eye"])
        right_eye_path = os.path.join(self.root_dir, pair["right_eye"])
        label = pair["label"]
        if not os.path.exists(left_eye_path) or not os.path.exists(right_eye_path):
            raise FileNotFoundError(f"Images not found: {left_eye_path} or {right_eye_path}")
        left_eye = Image.open(left_eye_path).convert("RGB")
        right_eye = Image.open(right_eye_path).convert("RGB")
        if self.transform:
            left_eye = self.transform(left_eye)
            right_eye = self.transform(right_eye)
        combined_image = torch.cat([left_eye, right_eye], dim=0)

        return combined_image, label, id

    def calculate_label_weights(self):
        group_1_df = self.df[self.df['group'] == 1]
        label_counts = group_1_df['label'].value_counts()
        label_0_count = label_counts.get(0, 0)
        label_1_count = label_counts.get(1, 0)
        total = label_0_count + label_1_count
        if total == 0:
            return 0, 0
        weight_0 = label_0_count / total
        weight_1 = label_1_count / total
        return weight_0, weight_1

    def get_labels(self):
        return self.df['label'].values


