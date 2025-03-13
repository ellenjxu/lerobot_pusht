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

# Example usage
actions_path = 'actions.pkl'
state_path = 'states.pkl'
position_path = 'positions.txt'

def get_perceptron(dataset, epochs=10_000, learning_rate=0.0001):
    states = np.array([item['state'] for item in dataset])
    positions = np.array([item['position'] for item in dataset])
    X_train = torch.tensor(states, dtype=torch.float32)
    y_train = torch.tensor(positions, dtype=torch.float32)

    model = nn.Sequential(
            nn.Linear(X_train.shape[1], 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 20),   
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

    print(loss)
    torch.save(model.state_dict(), 'model.pt')
    return model

def load_model():
    model = nn.Sequential(
            nn.Linear(6, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 20),   
            nn.ReLU(),
            nn.Linear(20, 2),   
        )
    model.load_state_dict(torch.load('model.pt'))
    model.eval()
    return model

def interpolate(dataset, target_state, k=1):
    states = np.array([item['state'] for item in dataset])
    positions = np.array([item['position'] for item in dataset])

    distances = np.linalg.norm(states - target_state, axis=1)
    nearest_indices = np.argsort(distances)[:k]
    nearest_positions = positions[nearest_indices]
    nearest_distances = distances[nearest_indices]
    nearest_distances = np.clip(nearest_distances, a_min=1e-6, a_max=None)
    
    weights = (1 / nearest_distances)
    return np.average(nearest_positions, axis=0, weights=weights)

def interpolate_inverse(dataset, target_position, k=5):
    positions = np.array([item['position'] for item in dataset])
    states = np.array([item['state'] for item in dataset])
    
    distances = np.linalg.norm(positions - target_position, axis=1)
    nearest_indices = np.argsort(distances)[:k]
    nearest_states = states[nearest_indices]
    nearest_distances = np.clip(distances[nearest_indices], 1e-6, None)
    
    weights = 1 / nearest_distances
    return np.average(nearest_states, axis=0, weights=weights).tolist()

def forward_kinematics(dataset, joints, use_model=True, k = 1000) -> np.array:
    assert isinstance(joints,np.ndarray)
    assert joints.shape == (6,)
    if use_model:
        model = load_model()
        return model(joints).detach().numpy()
    else:
        return interpolate(dataset, joints, k)

def inverse_kinematics(dataset, point, use_cache, backward_cache=None, k=1000):
    assert isinstance(point, np.ndarray)
    assert point.shape == (2,)
    return interpolate_inverse(dataset, point, k)

    
if __name__ == '__main__':
    dataset = load_and_process_data(actions_path, state_path, position_path)
    dataset = dataset.filter(lambda row: not np.isnan(row['position'][0]) and not np.isnan(row['position'][1]))
    split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']
    model = load_model()

    eval_states = np.array([item['state'] for item in eval_dataset])
    print("Modle ")
    predicted_positions_model = model(torch.tensor(eval_states, dtype=torch.float32)).detach().numpy()
    predicted_positions_interp = np.array([interpolate(train_dataset, t, 5) for t in eval_states])
    actual_positions = eval_dataset['position']

    diff_interp = predicted_positions_interp - actual_positions
    diff_model = predicted_positions_model - actual_positions

    frobenius_norm_interp = np.linalg.norm(diff_interp)
    frobenius_norm_model = np.linalg.norm(diff_model)

    print(f"Frobenius Norm of Interpolation Difference: {frobenius_norm_interp}")
    print(f"Frobenius Norm of Model recovered_positionDifference: {frobenius_norm_model}")

    sample_idx = np.random.randint(len(eval_dataset))
    original_position = np.array(eval_dataset[sample_idx]['position'])
    noise = np.random.normal(0, 0.05, size=original_position.shape)
    distorted_position = original_position + noise

    estimated_joints = inverse_kinematics(train_dataset, distorted_position, use_cache=False, k=5)
    recovered_position = forward_kinematics(train_dataset, estimated_joints, use_model=True)

    print(f"Original Position: {original_position}")
