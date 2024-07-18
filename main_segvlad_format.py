
import argparse
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import vpr_models  # Assuming this module is available

class SimpleImageDataset(Dataset):
    def __init__(self, folder_path, image_size):
        self.image_paths = list(Path(folder_path).glob('*.png'))
        self.transform = transforms.Compose([
            transforms.Resize(image_size, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        return self.transform(image)

def extract_descriptors(model, dataloader, device):
    descriptors = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            desc = model(batch).cpu().numpy()
            descriptors.append(desc)
    return np.vstack(descriptors)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--database_folder', required=True)
    parser.add_argument('--queries_folder', required=True)
    parser.add_argument('--image_size', nargs=2, type=int, default=[480, 640])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--output_dir', default='output')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cosplace_config = ['cosplace', 'ResNet50', 2048]
    eigenplaces_config = ['eigenplaces', 'ResNet50', 2048]
    mixvpr_config = ['mixvpr', 'ResNet50', 4096]

    config_final = mixvpr_config#eigenplaces_config #cosplace_config

    method = config_final[0]
    backbone = config_final[1]
    descriptors_dimension = config_final[2]
    # method = 'cosplace'
    # backbone = 'ResNet50'
    # descriptors_dimension = 2048
    # model = vpr_models.get_model('cosplace', 'ResNet50', 2048).eval().to(device)
    model = vpr_models.get_model(method, backbone, descriptors_dimension).eval().to(device)
    output_dir = Path("/scratch/saishubodh/segments_data/VPAir/out/baseline_all") / method


    for folder in ['database', 'queries']:
        dataset = SimpleImageDataset(getattr(args, f'{folder}_folder'), tuple(args.image_size))
        dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)
        descriptors = extract_descriptors(model, dataloader, device)
        
        # output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        np.save(output_dir / f'{folder}_descriptors.npy', descriptors)
        print(f"Saved {folder} descriptors with shape {descriptors.shape}")

if __name__ == '__main__':
    main()