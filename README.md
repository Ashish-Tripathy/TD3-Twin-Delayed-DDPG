# Twin Delayed DDPG (TD3) Implementation

TD3 is an actor-critic model similar as in AC3 but is mainly used for continuous action predictions in robotics scenario. It extends DDPG with multiple improvements. Here we have twin critics which help to reduce the over-estimation of value function, the delayed updates of target and noise regularizations.

Below is the algorithm for TD3:



![alt ALGO](https://spinningup.openai.com/en/latest/_images/math/b7dfe8fa3a703b9657dcecb624c4457926e0ce8a.svg)



## Implementation steps:

### Step 1: Defining Replay memory:

A replay memory holds the experience of an agent while traversing through the environment. It observes the agents action and stores these observations. At any time step an observation can be defined by the agent's current state, next state, action it takes in the current state to move to next state and reward it earns from this step. Each observation is called as transition.

We define a class for Replay memory. Each object of replay memory has a storage array of max_size N (1e6 in my case). The storage array is filled up with tuples of transition first completely randomly based on random action steps of the agent, but as the agent keeps learning the environment, we fill the replay buffer with actions predicted by our RL model. A transition tuple also saves a done flag, a Boolean value which is set as 1 if the episode terminates after the action step.



![alt ALGO](https://i.imgur.com/l6IoD3h.png)

We define Add class function which fills Transitions in the storage array sequentially using append function. Once our array pointer reaches the index equal to the max_size, we re-write the storage array from the first position.

We also define Sample function in the Replay memory class. This function takes batch size as input. Based on this batch size, it finds equal number of random transition tuples from the storage array. 

```python
class ReplayBuffer(object):
  def __init__(self, max_size = 1e6):
    self.storage = []
    self.max_size = max_size
    self.ptr = 0
  
  def add(self, transition):
    if len(self.storage) == max_size:
      self.storage[int(self.ptr)] = transition
      self.ptr = (self.ptr+1)%max_size

  def sample(self, batch_size):
    ind = np.random.randint(0,len(self.storage), batch_size)
    batch_states, batch_next_states, batch_actions, \
    	batch_rewards, batch_dones = [],[],[],[],[]
    for i in ind:
      state, next_state, action, reward, done = self.storage[i]
      batch_states.append(np.array(state, copy=False))  
      batch_next_states.append(np.array(next_state, copy=False))
      batch_actions.append(np.array(action, copy=False))
      batch_rewards.append(np.array(reward, copy=False))
      batch_dones.append(np.array(done, copy=False))
    return np.array(batch_states),np.array(batch_next_states),np.array(batch_actions), \
    np.array(batch_rewards).reshape(-1,1),np.array(batch_dones).reshape(-1,1)
```





### Step 2: Defining the Model architecture:

1. Actor: It decides which action to take. It takes state as input. It essentially controls how the agent behaves by learning the optimal policy (policy-based). 
2. Critic: Critic evaluates the action predicted by actor by computing the value function.

![alt actor critic](https://i.imgur.com/TI8naMe.png)

To define the actor and critic, we make classes inheriting the Pytorch's nn.Module object. 

 ```python
class Actor(nn.Module):
  def __init__(self, state_dims, action_dim, max_action):
    super(Actor, self).__init__()
    self.layer_1 = nn.Linear(state_dims, 400)
    self.layer_2 = nn.Linear(400,300)
    self.layer_3 = nn.Linear(300, action_dim)
    self.max_action = max_action
  
  def forward(self, x):
    x = F.relu(self.layer_1(x))
    x = F.relu(self.layer_2(x))
    x = self.max_action * torch.tanh(self.layer_3(x))
    return x
 ```



DNN with 2 hidden layers of 400 and 300 nodes respectively. For predicting continuous action space (like predicting limb movement of a robot), we use tanh function, multiplying it with max_function value acquired from the environment, it returns continuous value clipped between max_action and -max_action. 

```python
class Critic(nn.Module):
  def __init__(self, state_dims, action_dim):
    super(Critic, self).__init__()
    self.layer_1  = nn.Linear(state_dims + action_dim, 400)
    self.layer_2 = nn.Linear(400, 300)
    self.layer_3 = nn.Linear(300,action_dim)

    def forward(self, x, u):
      xu = torch.cat([x,u],1)
      x1 = F.relu(self.layer_1(xu))
      x1 = F.relu(self.layer_2(x1))
      x1 = self.layer_3(x1)
      #forward propagation for second critic
      x2 = F.relu(self.layer_1(xu))
      x2 = F.relu(self.layer_2(x2))
      x2 = self.layer_3(x2)

      return x1,x2

    def Q1(self, x, u):
      xu = torch.cat([x,u],1)
      x1 = F.relu(self.layer_1(xu))
      x1 = F.relu(self.layer_2(x1))
      x1 = self.layer_3(x1)

      return x1
```



DNN with 2 hidden layers of 400 and 300 nodes respectively. the input dimension is defined as concatenation of action and state dimension, as a critic network takes state and action as inputs.

We build two versions of the each of actor and critic - Model and Target. Model is our usual DQN network, with actor and critic, critic tries to improve the actions predicted by our agent. Target network on the other hand is introduced to improve the Q-value targets temporarily so we don’t have a moving target to chase. It includes all the updates in the training. 

In TD3, we have two versions of each Critic Target and Critic Model. This is done to reduce over-estimation of value-function. 

We also define a function to update actor loss named as Q1.

So the entire architecture looks like:

![alt full_arch](https://i.imgur.com/40dicZM.png)



## Training Process

We create a TD3 class and initialise the variables.

```python
class TD3(object):
  def __init__(self, state_dims, action_dim, max_action):
    self.actor = Actor(state_dims, action_dim, max_action).to(device)
    self.actor_target = Actor(state_dims, action_dim, max_action).to(device)
    self.actor_target.load_state_dict(self.actor.state_dict)
    self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

    self.critic = Critic(state_dims, action_dim).to(device)
    self.critic_target = Critic(state_dims, action_dim).to(device)
    self.critic_target.load_state_dict(self.critic.state_dict)
    self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

    self.max_action = max_action
    
  def select_action(self, state):
    state = torch.Tensor(state.reshape(1,-1)).to(device)
    return self.actor(state).cpu().data.numpy().flatten()
```

We define a class function: select_action(). This function is takes state as an input, passes it to the actor class, which then calls its forward function to predict the action to be taken. Will see further how it is called in the training process.

### Step 4: Sampling Transitions

Sample from a batch of transitions from the Replay memory storage

![alt traning1](https://i.imgur.com/Nd5IdSl.png)

```python
  def train(self, replay_buffer, iterations, batch_size = 100, discount = 0.99,
            tau = 0.005, policy_noise = 0.2, noise_clip = 0.5, policy_freq = 2):
    
    for it in iterations:
      batch_states, batch_next_states, batch_actions, \
    	batch_rewards, batch_dones = replay_buffer.sample(batch_size)
      state = torch.Tensor(batch_states).to(device)
      next_state = torch.Tensor(batch_next_states).to(device)
      action = torch.Tensor(batch_actions).to(device)
      reward = torch.Tensor(batch_rewards).to(device)
      done = torch.Tensor(batch_dones).to(device)
```



### Step 5: Actor Target Predicts Next Action 

The actor target network uses the next state from the transition s' to predict the next action a'. It uses the forward() in actor class for prediction. 

![alt traning2](https://i.imgur.com/YN9fWkf.png)

```python 
		next_action = self.actor_target.forward(next_state)
```



### Step 6: Noise regularization on the predicted next action a'

Before sending a' to critic target networks, we add Gaussian noise to this next action a' and we clamp it in a range of values supported by the environment. So if we maximize our value estimates over actions with noise, we can expect our policies to be more stable and robust. It also introduces some sort of exploration to our agent. 

```python
        noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(device)
        noise = noise.clamp(-noise_clip, noice_clip)
        next_action = (next_action + noise).clamp(-self.max_action, self.max_action)
```



### Step 7: Q Value Estimation by Critic Targets

Predict Q values from both Critic target and take the minimum value

Both Critic targets take (s', a') as input and return Q values, Qt1(s', a') and Qt2(s', a') as outputs. 

```python
          target_Q1, target_Q2 = self.critic_target.forward(next_state, next_action)

          #Keep the minimum of these two Q-Values
          target_Q = torch.min(target_Q1, target_Q2)
```





### Step 8: Target value Computation

We use the target_Q computed in the last code block in the Bellman's equation as below:
$$
\begin{align*}
Qt = r + \gamma * min(Qt1, Qt2)
\end{align*}
$$


![alt training 4](https://i.imgur.com/1D9SRsQ.png)



```python
		   target_Q = reward + ((1-done) * discount * target_Q).detach()
```

The detach() is used to break the computational graphs and use the elements for further computation



### Step 9: Q value Estimation by Critic Models

Two critic models take (s, a) and return two Q-Values

![Alt training5](https://i.imgur.com/oa129cc.png)

```python
current_Q1, current_Q2 = self.critic.forward(state, action)
```

We call the critic class function forward() to predict the q-value taking the current state and current action as input.



### Step 10: Compute the Critic loss 

We compute the critic loss using the Q-values returned from the Critic model networks.
$$
Critic\ Loss = MSE(Q1(s,a),Qt) + MSE(Q2(s,a),Qt)
$$
![alt training6](https://i.imgur.com/hmhAElA.png)

```python
      critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
```



### Step 11: Update Critic Models

Backpropagate using Critic Loss and update the parameters of two Critic models.

![alt backprop](https://i.imgur.com/MtNQqjV.png)



```python
      self.critic_optimizer.zero_grad()
      critic_loss.backward()
      self.critic_optimizer.step()
```



### Step 12: Update Actor Model

Once every two iterations, we update our Actor Model by performing gradient ascent on the output of the first Critic Model.

![alt training8](https://i.imgur.com/KV9YnPx.png)



```python
      if it % policy_freq == 0:
        actor_loss = -(self.critic.Q1(state, self.actor(state)).mean())
        self.actor_optimizer.grad_zero()
        actor_loss.backward()
        self.actor_optimizer.step()
```



### Step 13: Update Actor Target

We soft update our actor target network using Polyak averaging. It is delayed and done after every two actor model update.

Polyak Averaging: 
$$
\theta' = \tau\theta + (1-\tau)\theta
$$


This way our target comes closer to our model. 

![alt training9](https://i.imgur.com/akToYxM.png)

```python
        for param, target_param in zip(self.actor.parameters(), \
                                       self.actor_target.parameters()):
          target_param.data.copy_(tau * param.data + (1-tau) * target_param.data)

```





### Step 14: Update Critic Target 

We soft update our critic target network along with our Actor Target using Polyak averaging.
$$
\phi' = \tau \phi + (1-\tau)\phi'
$$
![alt training final](https://i.imgur.com/fvX9eZK.png)

```python
        for param, target_param in zip(self.critic.parameters(),\
                                       self.critic_target.parameters()):
          target_param.data.copy_(tau * param.data + (1-tau) * target_param.data)
```

