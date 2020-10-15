import argparse
import os
import shutil

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Model')
    parser.add_argument('--query_img_name', default='/home/data/car/uncropped/008055.jpg', type=str,
                        help='query image name')
    parser.add_argument('--data_base', default='car_resnet50_512_4_0.03_0.5_data_base.pth',
                        type=str, help='queried database')
    parser.add_argument('--retrieval_num', default=8, type=int, help='retrieval number')

    opt = parser.parse_args()

    query_img_name, data_base_name, retrieval_num = opt.query_img_name, opt.data_base, opt.retrieval_num
    data_name = data_base_name.split('_')[0]

    data_base = torch.load('results/{}'.format(data_base_name))

    if query_img_name not in data_base['test_images']:
        raise FileNotFoundError('{} not found'.format(query_img_name))
    query_index = data_base['test_images'].index(query_img_name)
    query_image = Image.open(query_img_name).convert('RGB').resize((256, 256), resample=Image.BILINEAR)
    query_label = torch.tensor(data_base['test_labels'][query_index])
    query_feature = data_base['test_features'][query_index]

    gallery_images = data_base['{}_images'.format('test' if data_name != 'isc' else 'gallery')]
    gallery_labels = torch.tensor(data_base['{}_labels'.format('test' if data_name != 'isc' else 'gallery')])
    gallery_features = data_base['{}_features'.format('test' if data_name != 'isc' else 'gallery')]

    sim_matrix = []
    norm = torch.norm(query_feature, dim=-1)
    feature_weights = norm / torch.sum(norm, dim=0, keepdim=True)
    for i in range(feature_weights.size(0)):
        feature_vector = F.normalize(query_feature[i, :].unsqueeze(0), dim=-1)
        gallery_vector = F.normalize(gallery_features[:, i, :], dim=-1)
        sim_matrix.append(torch.mm(feature_vector, gallery_vector.t().contiguous()).squeeze())
    sim_matrix = torch.sum(torch.stack(sim_matrix, dim=-1) * feature_weights.unsqueeze(0), dim=-1)

    if data_name != 'isc':
        sim_matrix[query_index] = -1
    idx = sim_matrix.topk(k=retrieval_num, dim=-1)[1]

    result_path = 'results/{}'.format(query_img_name.split('/')[-1].split('.')[0])
    if os.path.exists(result_path):
        shutil.rmtree(result_path)
    os.mkdir(result_path)
    query_image.save('{}/query_img.jpg'.format(result_path))
    for num, index in enumerate(idx):
        retrieval_image = Image.open(gallery_images[index.item()]).convert('RGB') \
            .resize((256, 256), resample=Image.BILINEAR)
        draw = ImageDraw.Draw(retrieval_image)
        retrieval_label = gallery_labels[index.item()]
        retrieval_status = (retrieval_label == query_label).item()
        retrieval_dist = sim_matrix[index.item()].item()
        if retrieval_status:
            draw.rectangle((0, 0, 255, 255), outline='green', width=8)
        else:
            draw.rectangle((0, 0, 255, 255), outline='red', width=8)
        retrieval_image.save('{}/retrieval_img_{}_{}.jpg'.format(result_path, num + 1, '%.4f' % retrieval_dist))
