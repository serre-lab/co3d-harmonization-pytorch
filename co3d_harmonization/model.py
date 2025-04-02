import torch
import timm
import torch.nn as nn

class HarmonizationModel(torch.nn.Module):
    """
    Complete model for harmonization using gradient-based feature projection.
    
    This model uses a pre-trained timm backbone and adds a gradient projection module
    that analyzes input gradients with respect to the model's predictions. This allows
    the model to identify and project domain-specific features that influence predictions.

    Args:
        base_model_name (str): Name of the pre-trained timm model to use as base.
        num_classes (int): Number of output classes for classification.
        pretrained (bool): Whether to load the pre-trained weights for the base model.
        input_dim (int): Dimension of the input tensor for gradient projection.
        gradient_projection_dim (int, optional): Dimension of the gradient projection output. Defaults to 1.

    Attributes:
        base_model (timm.models): Pre-trained timm model for feature extraction and classification.
        gradient_projection (nn.Sequential): Neural network for projecting input gradients to a lower dimension.
    """

    def __init__(self, base_model_name, num_classes, pretrained, gradient_map_dim):
        super().__init__()
        
        self.base_model = timm.create_model(base_model_name, num_classes=num_classes, pretrained=pretrained)

        # Gradient projection layer to give the model more degrees of freedom to match humans
        self.gradient_projection = nn.Sequential(*[
            nn.Conv2d(gradient_map_dim, gradient_map_dim, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(gradient_map_dim, gradient_map_dim, kernel_size=1, padding=0)
        ])

    def forward(self, x):
        """
        Forward pass of the model with gradient-based feature projection.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: Tuple containing:
                - output (torch.Tensor): Classification logits from the base model
        """
        output = self.base_model(x)
        return output
    
    def saliency_projection(self, base_model_output):
        """
        Forward pass of the model with gradient-based feature projection.

        Args:
            base_model_output (torch.Tensor): Gradient map tensor from the base model.

        Returns:
            torch.Tensor: Projected gradients of the input with respect to predictions.
        """
        transformed_gradient_map = self.gradient_projection(base_model_output)
        return transformed_gradient_map
        # arg_output = torch.argmax(base_model_output, dim=1)  # What is the argmax of the output?
        # grad_selection = base_model_output[:, arg_output].mean()
        # grad_selection.backward()
        # input_grad = x.grad
        # input_projection = self.gradient_projection(input_grad)
        # cce = nn.CrossEntropyLoss()(output, labels)
        # harmonization_loss = nn.MSELoss()(input_projection, torch.zeros_like(input_projection))
        # loss = cce + alpha * harmonization_loss
    
class LinearProbeModel(torch.nn.Module):
    """
    Complete model for harmonization using gradient-based feature projection.
    
    This model uses a pre-trained timm backbone and adds a gradient projection module
    that analyzes input gradients with respect to the model's predictions. This allows
    the model to identify and project domain-specific features that influence predictions.

    Args:
        base_model_name (str): Name of the pre-trained timm model to use as base.
        num_classes (int): Number of output classes for classification.
        pretrained (bool): Whether to load the pre-trained weights for the base model.
        input_dim (int): Dimension of the input tensor for gradient projection.
        gradient_projection_dim (int, optional): Dimension of the gradient projection output. Defaults to 1.

    Attributes:
        base_model (timm.models): Pre-trained timm model for feature extraction and classification.
        gradient_projection (nn.Sequential): Neural network for projecting input gradients to a lower dimension.
    """

    def __init__(self, base_model_name, num_classes, pretrained, gradient_map_dim):
        super().__init__()
        
        self.base_model = timm.create_model(base_model_name, num_classes=0, pretrained=pretrained)

        # Linear Probe Layer
        self.linear_probe = nn.Linear(self.base_model.num_features, num_classes)

        # Gradient projection layer to give the model more degrees of freedom to match humans
        self.gradient_projection = nn.Sequential(*[
            nn.Conv2d(gradient_map_dim, gradient_map_dim, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(gradient_map_dim, gradient_map_dim, kernel_size=1, padding=0)
        ])

        # We freeze the base model
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # We make the linear layer trainable
        for param in self.linear_probe.parameters():
            param.requires_grad = True

        # We also make the linear layer trainable
        for param in self.gradient_projection.parameters():
            param.requires_grad = True

    def forward(self, x):
        """
        Forward pass of the model with gradient-based feature projection.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: Tuple containing:
                - output (torch.Tensor): Classification logits from the base model
        """
        x = self.base_model(x)
        output = self.linear_probe(x)

        return output
    
    def saliency_projection(self, base_model_output):
        """
        Forward pass of the model with gradient-based feature projection.

        Args:
            base_model_output (torch.Tensor): Gradient map tensor from the base model.

        Returns:
            torch.Tensor: Projected gradients of the input with respect to predictions.
        """
        transformed_gradient_map = self.gradient_projection(base_model_output)
        return transformed_gradient_map