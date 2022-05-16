import torch
import os

from Networks import netvlad, superpoint, superglue, ultrapoint

if __name__ == "__main__":
    assert torch.cuda.is_available()
    device = 'cuda'

    if not os.path.exists("models/SuperPoint_300.pt"):
        model = superpoint.SuperPoint(nms_radius=3, max_keypoints=300).eval().to(device)
        scripted_module = torch.jit.script(model)
        scripted_module.save("models/SuperPoint_300.pt")
        print("SuperPoint_300 Converted")
    else:
        print("SuperPoint_300 Exist")

    if not os.path.exists("models/SuperGlue_outdoor.pt"):
        model = superglue.SuperGlue(weights='outdoor', sinkhorn_iterations=50).eval().to(device)
        scripted_module = torch.jit.script(model)
        scripted_module.save("models/SuperGlue_outdoor.pt")
        print("SuperGlue_outdoor Converted")
    else:
        print("SuperGlue_outdoor Exist")

    if not os.path.exists("models/NetVLAD.pt"):
        model = netvlad.NetVLAD().eval().to(device)
        scripted_module = torch.jit.script(model)
        scripted_module.save("models/NetVLAD.pt")
        print("NetVLAD Converted")
    else:
        print("NetVLAD Exist")

    if not os.path.exists("models/UltraPoint.pt"):
        model = ultrapoint.UltraPoint().eval().to(device)
        scripted_module = torch.jit.script(model)
        scripted_module.save("models/UltraPoint.pt")
        print("UltraPoint Converted")
    else:
        print("UltraPoint Exist")