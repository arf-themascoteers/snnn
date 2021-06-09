import matplotlib.pyplot as plt
import random
import math

def celsius_to_fahrenheit(celsius):
    return (celsius * 9 / 5) + 32



LEARNING_RATE = 0.0001
celsius_values = range(-50,50)
SAMPLE_SIZE = len(celsius_values)
fahrenheit_values = []

for i in range(SAMPLE_SIZE):
    fahrenheit_values.insert(i, celsius_to_fahrenheit(celsius_values[i]))

weight = random.random()
bias = random.random()


def forward():
    output = []

    for i in range(SAMPLE_SIZE):
        output.insert(i, celsius_values[i] * weight + bias)

    cumulative_loss = 0

    for i in range(SAMPLE_SIZE):
        cumulative_loss = cumulative_loss + math.pow(output[i] - fahrenheit_values[i], 2)

    loss = cumulative_loss / SAMPLE_SIZE
    return output, loss


def backward(output):
    grad_output = []
    for i in range(SAMPLE_SIZE):
        grad_output.insert(i, 2*(output[i] - fahrenheit_values[i])/SAMPLE_SIZE)

    grad_weight = 0
    for i in range(SAMPLE_SIZE):
        grad_weight = grad_weight + output[i] * grad_output[i]

    grad_bias = sum(grad_output)

    return grad_weight, grad_bias


def update_parameters(grad_weight, grad_bias):
    global weight, bias

    descent_grad_weight = LEARNING_RATE * grad_weight
    descent_grad_bias = LEARNING_RATE * grad_bias

    weight = weight - descent_grad_weight
    bias = bias - descent_grad_bias


def train():
    for t in range(10000):
        output, loss = forward()
        grad_weight, grad_bias = backward(output)
        update_parameters(grad_weight, grad_bias)


def get_predicted_fahrenheit_values():
    prediction, loss = forward()
    return prediction


predicted_before_training = get_predicted_fahrenheit_values()
train()
predicted_after_training = get_predicted_fahrenheit_values()
fig, ax = plt.subplots()
ax.plot(celsius_values, fahrenheit_values, '-b', label='Actual')
ax.plot(celsius_values, predicted_before_training, '--r', label='Predicted before Training')
ax.plot(celsius_values, predicted_after_training, '--g', label='Predicted after Training')
plt.xlabel("Celsius")
plt.ylabel("Fahrenheit")
ax.legend(frameon=False)
plt.show()
