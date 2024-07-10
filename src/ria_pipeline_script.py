import numpy as np
from numpy.random import choice
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import csv as csv
from pykeen.triples import TriplesFactory
from pykeen.predict import predict_target
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_auc_score
from numpy.random import choice
from pykeen.pipeline import pipeline
import time
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.special import softmax
import sys

# Intracranial aneurysm ruptures prediction

# 1. KG Generation

def create_good(id, graph, train=True):
    state_2 = choose_state_2([0.056683716632900685, 0.6085446490193023, 0.2008064875214284, 0.06700692388182648, 0.06695822294454215])
    state_1 = choose_state_1([0.65, 0.175, 0.175])
    return add_patient(id, graph, train, "Home", state_1, state_2)

def create_medium(id, graph, train=True):
    state_2 = choose_state_2([0.4530133333333334, 0.20514666666666664, 0.20552666666666664, 0.06801999999999998, 0.06829333333333333])
    state_1 = choose_state_1([0.5, 0.375, 0.125])
    return add_patient(id, graph, train, "Reabilitation", state_1, state_2)

def create_bad(id, graph, train=True):
    state_2 = choose_state_2([0.04658203264455139, 0.1895982977994202, 0.5719793793822382, 0.09579876333414254, 0.0960415268396477])
    state_1 = choose_state_1([0.35, 0.575, 0.075])
    return add_patient(id, graph, train, "Death", state_1, state_2)

def choose_state_2(probs):
    states = ["Clipping", "Coiling", "FlowDiverter", "Occlusion", "Web"]
    return choice(states, 1, p=probs)[0]

def choose_state_1(probs):
    states = ["Emergency", "SAMU", "Transfer"]
    return choice(states, 1, p=probs)[0]

def add_patient(id, graph, train, output, state_1, state_2):
    if train: 
        graph += f"""P{id}\thasOutput\t{output}
{output}\toutput\tP{id}
"""
    graph += f"""{state_1}\tstate\tP{id}
{state_2}\tstate\tP{id}
"""
    return graph

functions = [create_good, create_bad, create_medium]

# Train (1000 patients)

graph = ""
y_train = np.array([choice(range(3), 1, p=[0.730540, 0.074806, 0.194654])[0] for _ in range(int(1000))])

for i, f in enumerate(y_train):
    graph = functions[f](i, graph)

Counter(y_train)

# Test (200 patients)

y_test = np.array([choice(range(3), 1, p=[0.730540, 0.074806, 0.194654])[0] for _ in range(int(200))])

for i, f in enumerate(y_test):
    graph = functions[f](i + 1000, graph, train=False)

Counter(y_test)

with open("kg_simple.xml", "w") as file:
    file.write(graph)

# 2. Model training

tf = TriplesFactory.from_path("kg_simple.xml")

model_name = sys.argv[1] if len(sys.argv) > 1 else 'TransE'
dataset = 'RIA'
embedding_dim = int(sys.argv[2]) if len(sys.argv) > 1 else 20
epochs = int(sys.argv[3]) if len(sys.argv) > 1 else 100

result = pipeline(
    model=model_name,       
    training=tf,
    testing=tf,     
    model_kwargs=dict(
        embedding_dim=embedding_dim,         
        loss="softplus", 
    ),  
    optimizer_kwargs=dict(
        lr=0.001,
        weight_decay=1e-4,
    ),  
    training_kwargs=dict(
        num_epochs=epochs, 
        use_tqdm_batch=False,
    ),  
    training_loop='sLCWA',
    negative_sampler='basic',
    device='cpu',
    use_tqdm=False,   
)

# plot loss
loss_plot = result.plot_losses()
loss_plot.figure.savefig(f'loss_{model_name}_{embedding_dim}_{epochs}_{time.strftime("%Y%m%d-%H%M%S")}.png',dpi=600)

# 3. Predict outcomes

preds = []
for i in range(len(y_test)):
    pred = predict_target(
            model=result.model,
            head=f"P{1000 + i}",
            relation="hasOutput",
            triples_factory=tf
        )
    preds += [pred]

y_pred = np.array([
    np.argmin(
        [pred.df.reset_index(drop=True)['tail_label'][pred.df.reset_index(drop=True)['tail_label'] == output].index[0] for output in ["Home", "Death", "Reabilitation"]]
    )
    for pred in preds
    ])

# 4. Predictions evaluation

print("EVALUATION")
print("Hits@1", round(result.get_metric('hits_at_1'), 2))
print("Hits@3", round(result.get_metric('hits_at_3'), 2))
print("Hits@5", round(result.get_metric('hits_at_5'), 2))
print("Hits@10", round(result.get_metric('hits_at_10'), 2))

matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=["Back2Home", "Death", "Reabilitation"])
disp.plot()
plt.savefig(f'confusion_{model_name}_{embedding_dim}_{epochs}_{time.strftime("%Y%m%d-%H%M%S")}.png')
plt.show()

print(classification_report(y_test, y_pred))

out_scores = np.array([
    softmax(
        [pred.df.reset_index(drop=True)['score'][pred.df.reset_index(drop=True)['tail_label'] == output].iloc[0] for output in ["Home", "Death", "Reabilitation"]]
    )
    for pred in preds
    ])

auc = round(roc_auc_score(y_test, out_scores, multi_class='ovr'), 2)
print("AUC ROC:", auc)

# 5. Create embedings

model = result.model

entity_embedding_tensor = model.entity_representations[0](indices=None).cpu()
relation_embedding_tensor = model.relation_representations[0](indices=None).cpu()

# 6. Show embeddigns

colors = [['g', 'r', 'b'][i] for i in y_train]
labels = [i for i in y_train]
patients_names = [f"P{i}" for i in range(1000)]
train_pos = entity_embedding_tensor[tf.entities_to_ids(patients_names)]
train_pos = train_pos.detach().numpy()

out_colors = ['g', 'r', 'b']
outcomes_names = ["Home", "Death", "Reabilitation"]
out_pos = entity_embedding_tensor[tf.entities_to_ids(outcomes_names)]
out_pos = out_pos.detach().numpy()

pca = PCA(n_components=2)
train_pos = pca.fit_transform(train_pos)
out_pos = pca.transform(out_pos)

print(f"{round(sum(pca.explained_variance_), 2)}% variance explained")

relation_names = ["hasOutput", "output"]
rel_pos = relation_embedding_tensor[tf.relations_to_ids(relation_names)]
rel_pos = rel_pos.detach().numpy()
origin = np.array([[0] * 2, [0] * 2])

plt.scatter(train_pos[:,0], train_pos[:,1], c=colors)
plt.scatter(out_pos[:,0], out_pos[:,1], s=200, marker='X', edgecolors=out_colors, facecolors=['w'] * len(out_colors))
plt.quiver(*origin, rel_pos[:,0], rel_pos[:,1], scale=8, color=['b', 'r'])
plt.show()

# 6.2 Other embeddings visualization

result.plot_er()

train_pos = entity_embedding_tensor[tf.entities_to_ids(patients_names)]
train_pos = train_pos.detach().numpy()

out_pos = entity_embedding_tensor[tf.entities_to_ids(outcomes_names)]
out_pos = out_pos.detach().numpy()

rel_pos = relation_embedding_tensor[tf.relations_to_ids(relation_names)]
rel_pos = rel_pos.detach().numpy()

points = np.concatenate((train_pos, out_pos, rel_pos))
tsne = TSNE(n_components=2)
points_tsne = tsne.fit_transform(points)

plt.scatter(points_tsne[:-5,0], points_tsne[:-5,1], c=colors)
plt.scatter(points_tsne[-5:-2,0], points_tsne[-5:-2,1], s=200, marker='X', edgecolors=out_colors, facecolors=['w'] * len(out_colors))
plt.quiver(*origin, points_tsne[-2:,0], points_tsne[-2:,1], scale=100, color=['b', 'r'])
plt.show()
