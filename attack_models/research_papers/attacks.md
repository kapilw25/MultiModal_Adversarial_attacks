# Implementing Adversarial Attacks: Black-Box vs White-Box

## 1. Projected Gradient Descent (PGD) Attack

### Black-Box (Inference) Implementation

1. **Load Substitute Model**: `model = models.resnet50(pretrained=True).to(device).eval()`
2. **Wrap Model in ART Classifier**: `classifier = PyTorchClassifier(model=model, loss=nn.CrossEntropyLoss(), input_shape=(3,224,224), nb_classes=1000)`
3. **Create PGD Attack**: `attack = ProjectedGradientDescent(classifier, norm=np.inf, eps=8/255, eps_step=2/255, max_iter=10)`
4. **Generate Adversarial Example**: `adv_image = attack.generate(x=preprocess(original_image))`
5. **Transfer to Target Model**: `target_prediction = target_model(postprocess(adv_image))`

## 2. Fast Gradient Sign Method (FGSM) Attack
### Black-Box (Inference) Implementation

1. **Load Substitute Model**: 
   ```python
   model = models.resnet50(pretrained=True)
   model.to(device).eval()
   ```

2. **Compute Gradient on Substitute**:
   ```python
   image_tensor = preprocess(image).unsqueeze(0).requires_grad_(True)
   output = model(image_tensor)
   loss = F.cross_entropy(output, target_label)
   loss.backward()
   ```

3. **Generate Adversarial Example**:
   ```python
   perturbation = epsilon * image_tensor.grad.sign()
   adv_image = torch.clamp(image_tensor + perturbation, 0, 1)
   ```

4. **Convert to Numpy/PIL**:
   ```python
   adv_image_np = adv_image.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0)
   adv_image_pil = Image.fromarray((adv_image_np * 255).astype(np.uint8))
   ```

5. **Test on Target Model**:
   ```python
   target_output = target_model(preprocess_target(adv_image_pil).unsqueeze(0))
   ```
## 1. Projected Gradient Descent (PGD) Attack
### White-Box (Finetuning) Implementation

1. **Access Model Directly**: `model.train(); optimizer = torch.optim.Adam(model.parameters())`
2. **Initialize Attack Parameters**: `epsilon=8/255; alpha=2/255; num_iter=10`
3. **PGD Attack Loop**:
   ```python
   delta = torch.zeros_like(x, requires_grad=True)
   for i in range(num_iter):
       loss = F.cross_entropy(model(x + delta), y)
       loss.backward()
       delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon, epsilon)
       delta.grad.zero_()
   ```
4. **Generate Adversarial Examples**: `x_adv = torch.clamp(x + delta, 0, 1)`
5. **Adversarial Training**: `loss = F.cross_entropy(model(x_adv), y); optimizer.zero_grad(); loss.backward(); optimizer.step()`

## 2. Fast Gradient Sign Method (FGSM) Attack



### White-Box (Finetuning) Implementation

1. **Setup Model and Data**:
   ```python
   model.train()
   optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
   x, y = next(iter(train_loader))
   x, y = x.to(device), y.to(device)
   ```

2. **Compute Gradients**:
   ```python
   x_input = x.clone().detach().requires_grad_(True)
   output = model(x_input)
   loss = F.cross_entropy(output, y)
   loss.backward()
   ```

3. **Generate FGSM Examples**:
   ```python
   perturbation = epsilon * x_input.grad.sign()
   x_adv = torch.clamp(x + perturbation, 0, 1)
   ```

4. **Train on Adversarial Examples**:
   ```python
   optimizer.zero_grad()
   adv_output = model(x_adv)
   adv_loss = F.cross_entropy(adv_output, y)
   adv_loss.backward()
   optimizer.step()
   ```

5. **Evaluate Robustness**:
   ```python
   model.eval()
   with torch.no_grad():
       clean_acc = (model(x).argmax(dim=1) == y).float().mean()
       adv_acc = (model(x_adv).argmax(dim=1) == y).float().mean()
   print(f"Clean accuracy: {clean_acc:.4f}, Adversarial accuracy: {adv_acc:.4f}")
   ```
