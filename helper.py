from datasets import load_dataset
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

def load_and_process_data(actions_path, state_path, position_path, dataset_name="ellen2imagine/koch_robot_data", data_dir='robot_cache'):
    # Load the dataset
    dataset = load_dataset(dataset_name, data_dir=data_dir, split='train', trust_remote_code=True)

    # Load actions and states from pickle files
    with open(actions_path, 'rb') as f:
        pickled_actions = pickle.load(f)

    with open(state_path, 'rb') as f:
        pickled_state = pickle.load(f)

    # Convert actions and states to list
    pickled_actions = [action.tolist() for action in pickled_actions]
    pickled_state = [state.tolist() for state in pickled_state]

    # Load the positions from position.txt
    positions = {}
    with open(position_path, 'r') as f:
        for line in f:
            idx, x, y = map(float, line.split(', '))
            positions[int(idx)] = (x, y)

    # Add actions and state columns
    dataset = dataset.add_column('action', pickled_actions)
    dataset = dataset.add_column('state', pickled_state)

    # Add position column with NaN for missing indices
    position_column = []
    for i in range(len(dataset)):
        if i in positions:
            position_column.append(positions[i])
        else:
            position_column.append([np.nan, np.nan])

    dataset = dataset.add_column('position', position_column)

    return dataset

def get_perceptron(dataset, epochs=10_000, learning_rate=0.0001):
    states = np.array([item['state'] for item in dataset])
    positions = np.array([item['position'] for item in dataset])
    X_train = torch.tensor(states, dtype=torch.float32)
    y_train = torch.tensor(positions, dtype=torch.float32)

    model = nn.Sequential(
            nn.Linear(X_train.shape[1], 20),
            nn.ReLU(),
            nn.Linear(20, 100),
            nn.ReLU(),
            nn.Linear(100, 20),   
            nn.ReLU(),
            nn.Linear(20, y_train.shape[1]),   
        )

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)

    for _ in range(epochs):
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (_ + 1) % 1000 == 0:
            print(f"Epoch {_ + 1}/{epochs}, Loss: {loss.item():.4f}")
    
    print(loss)
    torch.save(model.state_dict(), 'model_big.pt')
    return model

def get_inverse_kinematics_model(dataset, epochs=100_000, learning_rate=0.0001):
    """
    Train a neural network model for inverse kinematics (2D position to 6D joint angles)
    
    Args:
        dataset: Dataset containing 'position' and 'state' columns
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        
    Returns:
        Trained PyTorch model
    """
    # Extract positions (2D) as input and states (6D) as output
    positions = np.array([item['position'] for item in dataset])
    states = np.array([item['state'] for item in dataset])
    
    # Convert to PyTorch tensors
    X_train = torch.tensor(positions, dtype=torch.float32)  # 2D positions
    y_train = torch.tensor(states, dtype=torch.float32)     # 6D joint angles
    
    model = nn.Sequential(
        nn.Linear(X_train.shape[1], 20),  # Input: 2D position
        nn.ReLU(),
        nn.Linear(20, 100),
        nn.ReLU(),
        nn.Linear(100, 200),   
        nn.ReLU(),
        nn.Linear(200, y_train.shape[1])  # Output: 6D joint angles
    )
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)

    for epoch in range(epochs):
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 1000 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")
    
    print(f"Final loss: {loss.item():.4f}")
    torch.save(model.state_dict(), 'inverse_model_big.pt')
    return model


def load_inverse_model(model_name):
    model = nn.Sequential(
        nn.Linear(2, 20),    # Input: 2D position
        nn.ReLU(),
        nn.Linear(20, 100),
        nn.ReLU(),
        nn.Linear(100, 200),   
        nn.ReLU(),
        nn.Linear(200, 6)     # Output: 6D joint angles
    )
    model.load_state_dict(torch.load(model_name))
    model.eval()
    return model

def load_model(model_name):
    model = nn.Sequential(
        nn.Linear(6, 20),
        nn.ReLU(),
        nn.Linear(20, 100),
        nn.ReLU(),
        nn.Linear(100, 20),   
        nn.ReLU(),
        nn.Linear(20, 2),   
    )
    model.load_state_dict(torch.load(model_name))
    model.eval()
    return model



def interpolate(dataset, target_state, k=1):
    states = dataset['state']
    positions = dataset['position']
    distances = np.linalg.norm(states - target_state, axis=1)
    nearest_indices = np.argsort(distances)[:10] # k
    nearest_positions = positions[nearest_indices]
    nearest_distances = distances[nearest_indices]
    nearest_distances = np.clip(nearest_distances, a_min=1e-6, a_max=None)
    weights = (1 / nearest_distances) ** 0.2
    return np.average(nearest_positions, axis=0, weights=weights)

def interpolate_inverse(dataset, target_position, k=5):
    states = dataset['state']
    positions = dataset['position']
    distances = np.linalg.norm(positions - target_position, axis=1)
    nearest_indices = np.argsort(distances)[:10] # k
    nearest_states = states[nearest_indices]
    nearest_distances = np.clip(distances[nearest_indices], 1e-6, None)
    
    weights = (1 / nearest_distances) ** 0.2
    return np.average(nearest_states, axis=0, weights=weights)

#6D to 2D
def forward_kinematics(dataset, joints, model_name='model_big.pt', use_model=True, k = 1000) -> np.array:
    assert isinstance(joints,np.ndarray)
    assert np.all((-360 <= joints) & (joints <= 360))

    assert joints.shape == (6,)
    if use_model:
        model = load_model(model_name)
        return model(torch.tensor(joints, dtype=torch.float32)).detach().numpy()
    else:
        return interpolate(dataset, joints, k)

#2D to 6D
def inverse_kinematics(dataset, position, model_name='inverse_model_big.pt', use_model=True, k=1000):    
    assert isinstance(position, np.ndarray)
    assert 0 <= position[0] <= 25
    assert 0 <= position[1] <= 16
    assert position.shape == (2,)
    
    if use_model:
        model = load_inverse_model(model_name)
        return model(torch.tensor(position, dtype=torch.float32)).detach().numpy()
    else:
        return interpolate_inverse(dataset, position, k)

def get_dataset():
    actions_path = 'actions.pkl'
    state_path = 'states.pkl'
    position_path = 'positions.txt'
    dataset = load_and_process_data(actions_path, state_path, position_path)
    dataset = dataset.filter(lambda row: not np.isnan(row['position'][0]) and not np.isnan(row['position'][1]))
    dataset.set_format(type='numpy')
    return dataset

"""
During inference, we map from real -> sim -> real

Helper functions:forw

1. Image processing: image -> T (x,y,theta)
2. Forward kinematics: 6 joint angles -> calculate (x,y) in sim
3. Inverse kinematics: (x,y) -> 6 joint angles
"""

def process_image(img):
  """
  Convert image to (x,y,theta)
  """
  pass

# assume we have x,y,x_t,y_t,theta_t in the cartesian coord system
def real2sim_transform(x,y,x_t,y_t,theta_t):
    w,h = 512, 512
    scale = 32
    x, y = x*scale, h-y*scale
    x_t, y_t = x_t*scale, y_t*scale
    theta_t = -theta_t - np.pi/2 # same as theta_t+np.pi/2?

    # shift by 6 grids (since we crop the image)
    w_sq = 25.6
    x, x_t = x-6*w_sq, x_t-6*w_sq

    return x,y,x_t,y_t,theta_t

# only need to map agent x,y back
def sim2real_transform(x,y):
    w_sq = 25.6
    x = x+6*w_sq

    w,h = 512, 512
    scale = 32
    x, y = x/scale, (h-y)/scale

    return x,y

if __name__ == '__main__':
    dataset = get_dataset()
    # get_inverse_kinematics_model(dataset, epochs=20_000)
    joint = np.array([-20,  65,    90,    60,  210,    -7])

    predicted_joint = inverse_kinematics(dataset, forward_kinematics(dataset, joint, model_name='model_big.pt'), model_name='inverse_model_big.pt')
    print(predicted_joint)
    print(joint)
