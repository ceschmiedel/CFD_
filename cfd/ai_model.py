"""
Módulo de modelo substituto usando IA para simulações CFD rápidas
Implementa redes neurais para predição de campos de velocidade e pressão
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import pickle
import os

class CFDNet(nn.Module):
    """Rede neural para predição de campos CFD"""
    
    def __init__(self, input_size=4, hidden_sizes=[64, 128, 64], output_size=3):
        """
        Inicializa a rede neural
        
        Args:
            input_size (int): Tamanho da entrada (x, y, geometria_features)
            hidden_sizes (list): Tamanhos das camadas ocultas
            output_size (int): Tamanho da saída (u, v, p)
        """
        super(CFDNet, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Camadas ocultas
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_size = hidden_size
        
        # Camada de saída
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


class CFDAIModel:
    """Modelo de IA para simulações CFD"""
    
    def __init__(self, device=None):
        """
        Inicializa o modelo de IA
        
        Args:
            device: Dispositivo PyTorch (CPU/GPU)
        """
        self.device = device if device else torch.device('cpu')
        self.model = None
        self.scaler_input = StandardScaler()
        self.scaler_output = StandardScaler()
        self.is_trained = False
        
    def generate_training_data(self, n_samples=10000, domain_size=(4.0, 2.0)):
        """
        Gera dados sintéticos para treinamento
        
        Args:
            n_samples (int): Número de amostras
            domain_size (tuple): Tamanho do domínio (Lx, Ly)
            
        Returns:
            tuple: (X, y) dados de entrada e saída
        """
        Lx, Ly = domain_size
        
        # Gerar pontos aleatórios no domínio
        x = np.random.uniform(0, Lx, n_samples)
        y = np.random.uniform(0, Ly, n_samples)
        
        # Features geométricas simples (distância de obstáculos fictícios)
        # Simular alguns obstáculos circulares
        obstacles = [
            (Lx * 0.3, Ly * 0.5, 0.1),  # (x, y, raio)
            (Lx * 0.6, Ly * 0.3, 0.08),
            (Lx * 0.6, Ly * 0.7, 0.08)
        ]
        
        dist_features = []
        for obs_x, obs_y, obs_r in obstacles:
            dist = np.sqrt((x - obs_x)**2 + (y - obs_y)**2) - obs_r
            dist_features.append(np.maximum(dist, 0))  # Distância mínima = 0
        
        # Combinar features
        X = np.column_stack([x, y] + dist_features)
        
        # Gerar campos de velocidade e pressão sintéticos
        u = self._synthetic_velocity_u(x, y, obstacles, Lx, Ly)
        v = self._synthetic_velocity_v(x, y, obstacles, Lx, Ly)
        p = self._synthetic_pressure(x, y, obstacles, Lx, Ly)
        
        y = np.column_stack([u, v, p])
        
        return X, y
    
    def _synthetic_velocity_u(self, x, y, obstacles, Lx, Ly):
        """Gera campo de velocidade u sintético"""
        # Velocidade base (fluxo uniforme)
        u_base = 30.0 * np.ones_like(x)
        
        # Reduzir velocidade próximo aos obstáculos
        for obs_x, obs_y, obs_r in obstacles:
            dist = np.sqrt((x - obs_x)**2 + (y - obs_y)**2)
            influence = np.exp(-dist / (2 * obs_r))
            u_base *= (1 - 0.8 * influence)
        
        # Condições de contorno
        # Entrada: velocidade constante
        inlet_mask = x < 0.1 * Lx
        u_base[inlet_mask] = 30.0
        
        # Paredes: velocidade zero
        wall_mask = (y < 0.05 * Ly) | (y > 0.95 * Ly)
        u_base[wall_mask] = 0.0
        
        return u_base
    
    def _synthetic_velocity_v(self, x, y, obstacles, Lx, Ly):
        """Gera campo de velocidade v sintético"""
        # Velocidade base (pequena componente vertical)
        v_base = np.zeros_like(x)
        
        # Adicionar efeitos dos obstáculos
        for obs_x, obs_y, obs_r in obstacles:
            dist = np.sqrt((x - obs_x)**2 + (y - obs_y)**2)
            angle = np.arctan2(y - obs_y, x - obs_x)
            
            # Fluxo ao redor do obstáculo
            influence = np.exp(-dist / (3 * obs_r))
            v_base += 10.0 * influence * np.sin(angle)
        
        # Paredes: velocidade zero
        wall_mask = (y < 0.05 * Ly) | (y > 0.95 * Ly)
        v_base[wall_mask] = 0.0
        
        return v_base
    
    def _synthetic_pressure(self, x, y, obstacles, Lx, Ly):
        """Gera campo de pressão sintético"""
        # Pressão base (gradiente linear)
        p_base = 1000.0 * (1 - x / Lx)
        
        # Adicionar efeitos dos obstáculos
        for obs_x, obs_y, obs_r in obstacles:
            dist = np.sqrt((x - obs_x)**2 + (y - obs_y)**2)
            
            # Alta pressão na frente, baixa pressão atrás
            dx = x - obs_x
            pressure_effect = 500.0 * np.exp(-dist / obs_r)
            
            # Frente do obstáculo: alta pressão
            front_mask = dx < 0
            p_base[front_mask] += pressure_effect[front_mask]
            
            # Atrás do obstáculo: baixa pressão
            back_mask = dx > 0
            p_base[back_mask] -= 0.5 * pressure_effect[back_mask]
        
        return p_base
    
    def train(self, X=None, y=None, epochs=100, batch_size=256, learning_rate=0.001):
        """
        Treina o modelo de IA
        
        Args:
            X (np.array): Dados de entrada (se None, gera dados sintéticos)
            y (np.array): Dados de saída
            epochs (int): Número de épocas
            batch_size (int): Tamanho do batch
            learning_rate (float): Taxa de aprendizado
            
        Returns:
            list: Histórico de perdas
        """
        # Gerar dados se não fornecidos
        if X is None or y is None:
            print("Gerando dados sintéticos para treinamento...")
            X, y = self.generate_training_data()
        
        # Normalizar dados
        X_scaled = self.scaler_input.fit_transform(X)
        y_scaled = self.scaler_output.fit_transform(y)
        
        # Converter para tensores PyTorch
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.FloatTensor(y_scaled).to(self.device)
        
        # Criar modelo
        input_size = X.shape[1]
        output_size = y.shape[1]
        self.model = CFDNet(input_size=input_size, output_size=output_size).to(self.device)
        
        # Otimizador e função de perda
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # Dataset e DataLoader
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Treinamento
        loss_history = []
        self.model.train()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                
                # Forward pass
                predictions = self.model(batch_X)
                loss = criterion(predictions, batch_y)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            loss_history.append(avg_loss)
            
            if (epoch + 1) % 20 == 0:
                print(f"Época {epoch + 1}/{epochs}, Perda: {avg_loss:.6f}")
        
        self.is_trained = True
        print("Treinamento concluído!")
        
        return loss_history
    
    def predict(self, X):
        """
        Faz predições usando o modelo treinado
        
        Args:
            X (np.array): Dados de entrada
            
        Returns:
            np.array: Predições (u, v, p)
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Modelo não foi treinado ainda")
        
        self.model.eval()
        
        with torch.no_grad():
            # Normalizar entrada
            X_scaled = self.scaler_input.transform(X)
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            
            # Predição
            predictions_scaled = self.model(X_tensor).cpu().numpy()
            
            # Desnormalizar saída
            predictions = self.scaler_output.inverse_transform(predictions_scaled)
        
        return predictions
    
    def predict_field(self, domain_size=(4.0, 2.0), resolution=(80, 40), obstacles=None):
        """
        Prediz campos CFD em uma malha regular
        
        Args:
            domain_size (tuple): Tamanho do domínio
            resolution (tuple): Resolução da malha
            obstacles (list): Lista de obstáculos [(x, y, r), ...]
            
        Returns:
            dict: Campos preditos
        """
        Lx, Ly = domain_size
        nx, ny = resolution
        
        # Criar malha
        x = np.linspace(0, Lx, nx)
        y = np.linspace(0, Ly, ny)
        X_mesh, Y_mesh = np.meshgrid(x, y)
        
        # Achatar para predição
        x_flat = X_mesh.flatten()
        y_flat = Y_mesh.flatten()
        
        # Features geométricas
        if obstacles is None:
            # Usar obstáculos padrão
            obstacles = [
                (Lx * 0.3, Ly * 0.5, 0.1),
                (Lx * 0.6, Ly * 0.3, 0.08),
                (Lx * 0.6, Ly * 0.7, 0.08)
            ]
        
        dist_features = []
        for obs_x, obs_y, obs_r in obstacles:
            dist = np.sqrt((x_flat - obs_x)**2 + (y_flat - obs_y)**2) - obs_r
            dist_features.append(np.maximum(dist, 0))
        
        # Combinar features
        X_pred = np.column_stack([x_flat, y_flat] + dist_features)
        
        # Fazer predição
        predictions = self.predict(X_pred)
        
        # Reformatar para malha
        u_field = predictions[:, 0].reshape(ny, nx)
        v_field = predictions[:, 1].reshape(ny, nx)
        p_field = predictions[:, 2].reshape(ny, nx)
        
        return {
            'velocity_u': u_field,
            'velocity_v': v_field,
            'pressure': p_field,
            'velocity_magnitude': np.sqrt(u_field**2 + v_field**2),
            'mesh_x': X_mesh,
            'mesh_y': Y_mesh,
            'x_coords': x,
            'y_coords': y
        }
    
    def save_model(self, filepath):
        """Salva o modelo treinado"""
        if not self.is_trained:
            raise ValueError("Modelo não foi treinado ainda")
        
        model_data = {
            'model_state_dict': self.model.state_dict(),
            'scaler_input': self.scaler_input,
            'scaler_output': self.scaler_output,
            'model_architecture': {
                'input_size': self.model.network[0].in_features,
                'output_size': self.model.network[-1].out_features
            }
        }
        
        torch.save(model_data, filepath)
        print(f"Modelo salvo em: {filepath}")
    
    def load_model(self, filepath):
        """Carrega modelo treinado"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")
        
        model_data = torch.load(filepath, map_location=self.device)
        
        # Recriar modelo
        arch = model_data['model_architecture']
        self.model = CFDNet(
            input_size=arch['input_size'],
            output_size=arch['output_size']
        ).to(self.device)
        
        # Carregar pesos
        self.model.load_state_dict(model_data['model_state_dict'])
        
        # Carregar scalers
        self.scaler_input = model_data['scaler_input']
        self.scaler_output = model_data['scaler_output']
        
        self.is_trained = True
        print(f"Modelo carregado de: {filepath}")


def create_pretrained_model(device=None):
    """
    Cria e treina um modelo pré-configurado
    
    Args:
        device: Dispositivo PyTorch
        
    Returns:
        CFDAIModel: Modelo treinado
    """
    model = CFDAIModel(device=device)
    
    # Treinar com dados sintéticos
    print("Treinando modelo de IA...")
    loss_history = model.train(epochs=50)
    
    return model

