import torch
import torch.nn as nn
import torch.nn.functional as F


class NeRF(nn.Module):

    input_channel: int
    input_channel_views: int
    skips: list[int]
    use_viewdirs: bool
    pts_linears: nn.ModuleList
    feature_linear: nn.Linear | None = None
    alpha_linear: nn.Linear | None = None
    rgb_linear: nn.Linear | None = None
    output_linear: nn.Linear | None = None

    def __init__(
        self,
        D=8,
        W=256,
        input_channel=3,
        input_channel_views=3,
        output_channel=4,
        skips=[4],
        use_viewdirs=True,
    ):
        super(NeRF, self).__init__()
        self.input_channel = input_channel
        self.input_channel_views = input_channel_views
        self.skips = skips

        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_channel, W)]
            + [
                nn.Linear(W, W) if i not in skips else nn.Linear(W + input_channel, W)
                for i in range(D - 1)
            ]
        )

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_channel)

    def forward(self, x: torch.Tensor)-> torch.Tensor:
        inputs_pts, input_views = torch.split(x, [self.input_channel, self.input_channel_views], dim=-1)

        h : torch.Tensor = inputs_pts

        for i, l in enumerate(self.pts_linears):
            h = F.relu(l(h))
            if i in self.skips:
                h = torch.cat([inputs_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h) #type: ignore
            feature = self.feature_linear(h) #type: ignore
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = F.relu(l(h))

            rgb = self.rgb_linear(h) #type: ignore
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h) #type: ignore

        return outputs

