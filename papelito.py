# import torch

# embeddings=torch.randn((54,128))

# amount_embeddings=embeddings.size(0)

# labels=torch.zeros(size=amount_embeddings)

# print(labels)

import torch
from torch.nn import functional as F
from sklearn.metrics import accuracy_score

def get_embeddings_and_labels(dataloader, model, device):
    model.eval()
    embeddings_list = []
    labels_list = []
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            embeddings = model(X)
            embeddings_list.append(embeddings.cpu())
            labels_list.append(y.cpu())
    
    all_embeddings = torch.cat(embeddings_list)
    all_labels = torch.cat(labels_list)
    
    return all_embeddings, all_labels

def calculate_accuracy(embeddings, labels, threshold=0.5):
    num_samples = embeddings.size(0)
    predicted_labels = torch.zeros(num_samples, dtype=torch.long)
    
    for i in range(num_samples):
        distances = F.pairwise_distance(embeddings[i].unsqueeze(0), embeddings)
        closest_idx = torch.argmin(distances)
        predicted_labels[i] = labels[closest_idx]
    
    accuracy = accuracy_score(labels, predicted_labels)
    return accuracy

# Obtener incrustaciones y etiquetas del conjunto de prueba
test_embeddings, test_labels = get_embeddings_and_labels(test_loader, model_1, device)

# Calcular precisión en base a incrustaciones
accuracy = calculate_accuracy(test_embeddings, test_labels)
print(f'Test Accuracy: {accuracy:.4f}')