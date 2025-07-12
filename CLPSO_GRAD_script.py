import torch
import numpy as np
import torch.nn as nn
import time
from model import MiniTransformer

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def run_clpso(
    model_path,
    get_batch_fn,
    criterion,
    vocab_size,
    fine_tune_epochs=25,
    num_particles=30,
    w=0.5,
    c1=1.5,
    c2=1.5,
    bounds=0.1,
    p_threshold=0.05,
    gd_learning_rate=0.1,
    gd_weight_decay=0.01,
    num_grad_steps=1,
    num_eval_batches=5
):
    """
    Fine-tunes the head of a MiniTransformer model using a hybrid CLPSO-gradient algorithm.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Load Model and Freeze Body
    checkpoint = torch.load(model_path, map_location=device)
    model = MiniTransformer(vocab_size=checkpoint['vocab_size'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    for name, param in model.named_parameters():
        if 'head' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    #Initialize PSO Parameters and Particles
    num_ftrs = model.head.in_features
    total_params = (num_ftrs * vocab_size) + vocab_size

    #Start from the pre-trained weights
    initial_weights = np.concatenate([
        model.head.weight.data.view(-1).cpu().numpy(),
        model.head.bias.data.cpu().numpy()
    ])

    particles = [initial_weights + np.random.uniform(-0.01, 0.01, total_params) for _ in range(num_particles)]
    velocities = [np.zeros(total_params) for _ in range(num_particles)]
    personal_best_positions = [np.copy(p) for p in particles]
    personal_best_scores = [float('inf')] * num_particles

    #Evaluate the initial position
    model.eval()
    initial_loss = 0.0
    with torch.no_grad():
        for _ in range(num_eval_batches):
            xb, yb = get_batch_fn('val')
            outputs = model(xb)
            loss = criterion(outputs.view(-1, vocab_size), yb.view(-1))
            initial_loss += loss.item()
    
    global_best_score = initial_loss / num_eval_batches
    global_best_position = np.copy(initial_weights)
    print(f"Initial Global Best Fitness (from pre-trained head): {global_best_score:.4f}")


    #Setup Optimizer and Tracking
    optimizer = torch.optim.SGD(model.head.parameters(), lr=gd_learning_rate, weight_decay=gd_weight_decay)
    early_stopping = EarlyStopping(patience=5)
    
    epoch_losses = []
    epoch_precisions = []
    
    #Main Optimization Loop
    for epoch in range(fine_tune_epochs):
        start_time = time.time()
        print(f"Epoch {epoch + 1}/{fine_tune_epochs}")

        #PSO and Gradient Update Loop
        for i in range(num_particles):
            #PSO Velocity and Position Update
            r1, r2 = np.random.rand(total_params), np.random.rand(total_params)
            learning_probability = np.random.rand(total_params)
            
            for d in range(total_params):
                if learning_probability[d] < p_threshold:
                    selected_particle_idx = np.random.choice(num_particles)
                    learning_source = personal_best_positions[selected_particle_idx][d]
                else:
                    learning_source = personal_best_positions[i][d]

                velocities[i][d] = (w * velocities[i][d] +
                                    c1 * r1[d] * (learning_source - particles[i][d]) +
                                    c2 * r2[d] * (global_best_position[d] - particles[i][d]))
                velocities[i][d] = np.clip(velocities[i][d], -bounds, bounds)

            particles[i] += velocities[i]
            particles[i] = np.clip(particles[i], -1, 1)
            
            #Gradient Refinement Step
            particle_tensor = torch.from_numpy(particles[i]).float().to(device)
            weight_part = particle_tensor[:-vocab_size].view_as(model.head.weight)
            bias_part = particle_tensor[-vocab_size:]
            model.head.weight.data.copy_(weight_part)
            model.head.bias.data.copy_(bias_part)

            optimizer.zero_grad()
            model.train()
            for _ in range(num_grad_steps):
                xb, yb = get_batch_fn('train')
                outputs = model(xb)
                loss = criterion(outputs.view(-1, vocab_size), yb.view(-1))
                loss.backward()
            optimizer.step()

            #Update particle position with the new gradient-refined weights
            with torch.no_grad():
                particles[i] = np.concatenate([
                    model.head.weight.data.view(-1).cpu().numpy(),
                    model.head.bias.data.cpu().numpy()
                ])

        #Fitness Evaluation Loop
        current_fitnesses = [0.0] * num_particles
        for i in range(num_particles):
            particle_tensor = torch.from_numpy(particles[i]).float().to(device)
            weight_part = particle_tensor[:-vocab_size].view_as(model.head.weight)
            bias_part = particle_tensor[-vocab_size:]
            model.head.weight.data.copy_(weight_part)
            model.head.bias.data.copy_(bias_part)

            model.eval()
            total_loss = 0.0
            with torch.no_grad():
                for _ in range(num_eval_batches):
                    xb, yb = get_batch_fn('val')
                    outputs = model(xb)
                    loss = criterion(outputs.view(-1, vocab_size), yb.view(-1))
                    total_loss += loss.item()
            current_fitnesses[i] = total_loss / num_eval_batches

        #Update Personal and Global Bests
        for i, fitness in enumerate(current_fitnesses):
            if fitness < personal_best_scores[i]:
                personal_best_scores[i] = fitness
                personal_best_positions[i] = particles[i].copy()
            if fitness < global_best_score:
                global_best_score = fitness
                global_best_position = particles[i].copy()
                print(f"New global best fitness: {global_best_score:.4f}")

        #Load the best weights found in this epoch into the model
        with torch.no_grad():
            global_best_tensor = torch.from_numpy(global_best_position).float().to(device)
            weight_part = global_best_tensor[:-vocab_size].view_as(model.head.weight)
            bias_part = global_best_tensor[-vocab_size:]
            model.head.weight.data.copy_(weight_part)
            model.head.bias.data.copy_(bias_part)

        model.eval()
        val_loss = 0.0
        total_correct = 0
        total_tokens = 0
        with torch.no_grad():
            for _ in range(num_eval_batches):
                xb, yb = get_batch_fn('val')
                outputs = model(xb)
                loss = criterion(outputs.view(-1, vocab_size), yb.view(-1))
                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=-1)
                total_correct += (preds == yb).float().sum().item()
                total_tokens += yb.numel()

        val_loss /= num_eval_batches
        precision = total_correct / total_tokens
        epoch_losses.append(val_loss)
        epoch_precisions.append(precision)

        print(f"Epoch {epoch + 1}/{fine_tune_epochs} - Val Loss: {val_loss:.4f}, Val Acc: {precision*100:.2f}%")
        
        end_time = time.time()
        print(f"Epoch completed in {end_time - start_time:.2f}s. Best Global Fitness: {global_best_score:.4f}")

        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    print(f"\nOptimization completed. Final Best Global Fitness: {global_best_score:.4f}")
    
    #Ensure the model has the best weights before returning
    with torch.no_grad():
        global_best_tensor = torch.from_numpy(global_best_position).float().to(device)
        weight_part = global_best_tensor[:-vocab_size].view_as(model.head.weight)
        bias_part = global_best_tensor[-vocab_size:]
        model.head.weight.data.copy_(weight_part)
        model.head.bias.data.copy_(bias_part)
        
    return model, epoch_losses, epoch_precisions
