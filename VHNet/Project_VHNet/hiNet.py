import torch.nn as nn
from invBlock import INV_block
import torch
import config as c


class Hinet(nn.Module):

    def __init__(self, clip_len=c.clip_len):
        super(Hinet, self).__init__()
        self.clip_len = clip_len*2

        self.inv1 = INV_block()
        self.inv2 = INV_block()
        self.inv3 = INV_block()
        self.inv4 = INV_block()
        self.inv5 = INV_block()
        self.inv6 = INV_block()
        self.inv7 = INV_block()
        self.inv8 = INV_block()

        self.inv9 = INV_block()
        self.inv10 = INV_block()
        self.inv11 = INV_block()
        self.inv12 = INV_block()


        self.inv13 = INV_block()
        self.inv14 = INV_block()
        self.inv15 = INV_block()
        self.inv16 = INV_block()

    def forward(self, x, rev=False):

        if not rev:
            x = torch.cat(x.unbind(2), 1)  # b,c,n,h,w -> b,cn,h,w

            out = self.inv1(x)
            out = self.inv2(out)
            out = self.inv3(out)
            out = self.inv4(out)
            out = self.inv5(out)
            out = self.inv6(out)
            out = self.inv7(out)
            out = self.inv8(out)

            out = self.inv9(out)
            out = self.inv10(out)
            out = self.inv11(out)
            out = self.inv12(out)


            out = self.inv13(out)
            out = self.inv14(out)
            out = self.inv15(out)
            out = self.inv16(out)

            b, cn, h, w = out.shape
            out = out.view([b, self.clip_len, cn // self.clip_len, h, w])  # b,n,c,h,w
            out = out.transpose(1,2)
        else:

            x = torch.cat(x.unbind(2), 1)  # b,c,n,h,w -> b,cn,h,w


            ##1
            out = self.inv16(x, rev=True)
            out = self.inv15(out, rev=True)
            out = self.inv14(out, rev=True)
            out = self.inv13(out, rev=True)
            out = self.inv12(out, rev=True)
            ##2
            # out = self.inv12(x, rev=True)

            out = self.inv11(out, rev=True)
            out = self.inv10(out, rev=True)
            out = self.inv9(out, rev=True)

            out = self.inv8(out, rev=True)
            out = self.inv7(out, rev=True)
            out = self.inv6(out, rev=True)
            out = self.inv5(out, rev=True)
            out = self.inv4(out, rev=True)
            out = self.inv3(out, rev=True)
            out = self.inv2(out, rev=True)
            out = self.inv1(out, rev=True)

            b, cn, h, w = out.shape
            out = out.view([b, self.clip_len, cn // self.clip_len, h, w])  # b,n,c,h,w
            out = out.transpose(1,2)

        return out
