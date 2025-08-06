from scendi.score import ScendiEvaluator
from scendi.datasets.ImageFilesDataset import ImageFilesDataset


sigma = 3.5 # Gaussian kernel bandwidth parameter
fe = 'clip'

result_name = 'your_result_name'

img_pth = 'path_to_images'
text_pth = 'path_to_text.txt'

with open(text_pth, 'r') as f:
    prompts = f.readlines()
image_dataset = ImageFilesDataset(img_pth, name=result_name, extension='png')

num_samples = len(prompts)
assert len(prompts) == len(image_dataset.files)


scendi = ScendiEvaluator(logger_path='./logs', batchsize=64, sigma=sigma, num_samples=num_samples, result_name=result_name, rff_dim=2500, save_visuals_path=f'visuals_{result_name}')
scendi.set_schur_feature_extractor(fe, save_path='./save')    

# Cluster Results
scendi.scendi_clustering_of_dataset(prompts, image_dataset)

# Get Scendi Score
score = scendi.scendi_score(prompts, image_dataset)
print(score)

    
    
    





