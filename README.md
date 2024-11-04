## knapsack
Solutions to knapsack problem variations.

Currently, only solutions for knapsack 0/1 are available.

### Knapsack 0/1
Contains the following solutions:
+ #### Deterministic Recursive
  <pre>
  Time Complexity  : O( 2<sup>(Number of items)</sup> )
  Space Complexity : O( (Number of items)*(Recursive function frame size) )
  </pre>
+ #### Deterministic Recursive with 2D buffer
  <pre>
  Time Complexity  : O( (Number of items)*(Total knapsack space) )
  Space Complexity : O( (Number of items)*(Total knapsack space) )
  </pre>
+ #### Deterministic Iterative with 2D buffer
  <pre>
  Time Complexity  : O( (Number of items)*(Total knapsack space) )
  Space Complexity : O( (Number of items)*(Total knapsack space) )
  </pre>
+ #### Deterministic Iterative with 1D buffer
  <pre>
  Time Complexity  : O( (Number of items)*(Total knapsack space) )
  Space Complexity : O( (Total knapsack space) )
  </pre>
+ #### Stochastic with Simulated Annealing
  <pre>
  Time Complexity  : O( (Number of iterations in Simulated Annealing) )
  Space Complexity : O( (Number of bits needed to store the state of included items) )
  </pre>
