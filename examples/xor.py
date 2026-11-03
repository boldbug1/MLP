from minigrad.nn import MLP

xs = [
    [1, 0, 1, 0, 0],  # sum=2 -> 0
    [0, 1, 0, 1, 1],  # sum=3 -> 1
    [0, 0, 0, 0, 0],  # sum=0 -> 0
    [1, 1, 1, 0, 0],  # sum=3 -> 1
    [1, 0, 0, 0, 0],  # sum=1 -> 1
    [1, 1, 1, 1, 0],  # sum=4 -> 0
    [0, 1, 1, 0, 0],  # sum=2 -> 0
    [1, 1, 1, 1, 1],  # sum=5 -> 1
    [0, 0, 1, 0, 1],  # sum=2 -> 0
    [0, 1, 0, 0, 0],  # sum=1 -> 1
]
ys = [-1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0]

model = MLP(5,[8,8,1])

epochs = 150
lr = 0.05

for epoch in range(epochs):
    ypred = [model(x) for x in xs]
    loss = sum((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred)) * (1.0/len(ys))

    for p in model.parameters():
        p.grad = 0.0
    loss.backward()

    for p in model.parameters():
        p.data += -lr * p.grad
    print(epoch,loss)

for x, y in zip(xs, ys):
    pred = model(x)
    print(f"Target: {y}, Prediction: {pred.data:.4f}")