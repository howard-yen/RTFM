from PIL import Image
import pickle
import torch
from torchvision import transforms
import os
import matplotlib.pyplot as plt

ASSET_DIR = "./assets"
RESOLUTION = 32
DEBUG = False

if __name__ == "__main__":
    ret = {}
    to_tensor = transforms.ToTensor()
    for filename in os.listdir(ASSET_DIR):
        print(f"Processing {filename}")
        if not filename.endswith(".png"):
            continue
        entity_name = filename.split(".")[0]

        with Image.open(os.path.join(ASSET_DIR, filename)).convert("RGBA") as img:
            height, width = img.size
            if height > width:
                new_img = img.resize((RESOLUTION, (int)(RESOLUTION/height*width)), Image.ANTIALIAS)
            else:
                new_img = img.resize(((int)(RESOLUTION/width*height), RESOLUTION), Image.ANTIALIAS)

            img_tensor = to_tensor(new_img).permute(1,2,0)
        
        ret[entity_name] = img_tensor

        # for debugging
        if DEBUG:
            plt.imshow(img_tensor)
            plt.show()
            plt.savefig(filename)
            plt.clf()
            import pdb; pdb.set_trace()
    
    print("Dumping pickle")
    with open("img_tensors.p", "wb") as f:
        pickle.dump(ret, f)
    
    with open("img_tensors.p", "rb") as f:
        read = pickle.load(f)

    for k, v in ret.items():
        assert k in read
        assert torch.allclose(v, read[k])
    print("Everything looks right!")