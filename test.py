import torch

# Tạo một tensor x và yêu cầu tính toán gradient của nó
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# Thực hiện một số phép toán trên x
y = x * 2
z = y

# Phép toán tiếp theo sẽ không yêu cầu tính toán gradient từ z
w = z.sum()

# Thực hiện backward pass
w.backward()

print(x.grad)  # Output: None, vì z đã được detach, do đó không có gradient được tính toán cho x
