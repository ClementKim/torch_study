import torch
import numpy as np

# 데이터로부터 직접 생성
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

# nunpy 배열로부터 생성
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
# 다른 텐서로부터 생성
x_ones = torch.ones_like(x_data)
print(f"Ones Tensor:\n{x_ones}\n")

x_rand = torch.rand_like(x_data, dtype=torch.float)
print(f"Random Tensor:\n{x_rand}\n")

shape = (3, 4,) # (행의 개수, 열의 개수)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor:\n{rand_tensor}\n")
print(f"Ones Tensor:\n{ones_tensor}\n")
print(f"Zeros Tensor:\n{zeros_tensor}\n")

tensor = torch.rand(3, 4)
print(f"shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on {tensor.device}")

print(tensor)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using {device} device")

tensor = tensor.to(device)

tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")

tensor[:,1] = 0

print(tensor, "\n")

t1 = torch.cat([tensor, tensor, tensor], dim = 0)
t2 = torch.cat([tensor, tensor, tensor], dim = 1)

print(t1, "\n")
print(t2, "\n")

## 산술연산(Arithmetic Operations)
# Matrix multiplication
# y1 = y2 = y3
# tensor.T returns tensor's trnaspose(전치)
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out = y3)

# element-wise product
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out = z3)

print(f"y1: {y1}\n")
print(f"y2: {y2}\n")
print(f"y3: {y3}\n")

print(f"z1: {z1}\n")
print(f"z2: {z2}\n")
print(f"z3: {z3}\n")

## 단일 요소(single element) 텐서
# 텐서의 모든 값을 하나로 집계(aggregate)하여 요소가 하나인 텐서의 경우, item()을 사용하여 Python 숫자 값으로 변환 가능
agg = tensor.sum()
agg_item = agg.item()
print(f"agg value: {agg}\n agg type: {type(agg)}")
print(f"agg_item value: {agg_item}\nagg item type: {type(agg_item)}")

## 바꿔치기(in place) 연산: 연산 결과를 피연산자(operand)에 저장하는 연산, _ 접미사를 가짐
# 기록이 즉시 삭제되어 도함수 계산에 문제 발생 가능으로 사용 권장 X
print(f"{tensor}\n")
tensor.add_(5)
print(tensor)

## NumPy 변환 (Bridge)
# cpu 상 텐서와 NumPy 배열은 메모리 공간을 공유하기 대문에, 하나를 변경하면 다른 하나도 변경
# tensor -> NumPy array
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

# NumPy array -> tensor
n = np.ones(5)
t = torch.from_numpy(n)

np.add(n, 1, out = n)
print(f"t: {t}")
print(f"n: {n}")
