#include <stdio.h>
#include <stdlib.h>

typedef unsigned int u32;
typedef float f32;

typedef struct Item Item;
struct Item {
	f32 weight;
	f32 value;
};

Item* generate_random_items(u32 n, u32 seed, f32 max_item_weight, f32 max_item_value) {
	srand(seed);
	Item* items = malloc(n * sizeof(Item));
	if(items) {
		for(u32 i = 0; i < n; ++i) {
			items[i].weight = ((f32)rand()/(f32)RAND_MAX * max_item_weight);
			items[i].value = ((f32)rand()/(f32)RAND_MAX * max_item_value);
		}
		return items;
	}
	else {
		return 0;
	}
}

void print_items(Item* items, u32 n) {
	for(u32 i = 0; i < n; ++i) {
		printf("( %9.6lf, %9.6lf )\n", items[i].weight, items[i].value);
	}
}

// =========================================================================================================
// MOST BASIC RECURSIVE SOLUTION
//
// Time complexity  : O(2^n)
// Space complexity : O(2^n * (stack frame size)) for total used memory
//                    O(n * (stack frame size))   for memory used at arbitrary time
// =========================================================================================================

// Returns the maximum value that can be achieved while not exceeding knapsack space.
// Also returns maximum reached weight.

// This algorithm moves recursively from the last item in a greedy fashion ie. when
// deciding whether to include current item or not, it will always take the route
// that gives it maximum value at the current step (local maximum).

// This is not efficient solution, since it does redundant computations.
// It struggles even with small sample size that are not trivial (like n = 50, knapsack_space = 500).
// This is because in that case, if items can't reduce space to zero, algorithm will recurse
// all the way down until n = 0 (which is horrible for even a small n).

// This solution does not require additional explicit space for storing computation information,
// but that is of little relevancy when it eats stack with function stack frames.
// For any larger input value, default stack size can get exceeded.

// Asymptotic time complexity is O(2^n) since we traverse all possible paths and this can lead to
// repetitions, where we calculate function value multiple times for the same input.

Item recursive_solution(Item* items, u32 n, f32 knapsack_space) {
	// If no more space or no more items.
	if(knapsack_space <= 0 || n == 0) {
		return (Item){0, 0};
	}
	
	// Exclude currently last item if its weight exceeds knapsack space and recurse further.
	if(items[n-1].weight > knapsack_space) {
		return recursive_solution(items, n-1, knapsack_space);
	}

	// Go through cases where we exclude currently last item and where we include it.
	// If we include it, then we count its value and weight and reduce current knapsack space.
	
	// We pick the path that gives maximum value for the current step (local computational maximum).
	Item value_excluded = recursive_solution(items, n-1, knapsack_space);
	Item value_included = recursive_solution(items, n-1, knapsack_space - items[n-1].weight);
	value_included.weight += items[n-1].weight;
	value_included.value  += items[n-1].value;
	
	if(value_excluded.value > value_included.value) {
		return value_excluded;
	}
	else {
		return value_included;
	}
}

// =========================================================================================================
// RECURSIVE SOLUTION WITH 2D TABLE (ITEM WEIGHTS AND KNAPSACK WEIGHT MUST BE INTEGERS IN THIS APPROACH)
//
// Time complexity  : O(n*total_knapsack_space)
// Space complexity : O(n*total_knapsack_space)
// =========================================================================================================

// This algorithm is the same as the most basic recursive one, but it uses additional buffer to
// avoid doing computations that were done previously. This is done by recognizing that different
// recursive calls to the function only differ in two arguments, "n" and "knapsack_space". That
// means that if we form a grid of all possible value pairs (n, knapsack_space), we can store all
// possible computation results of all recursive calls that we encounter, thus saving computation time
// if the same recursive calls happen later.

// It is irrelevant (from asymptotic point of view) how we organize this grid.
// Here, grid is just an array that is equivalent to the matrix with the first coordinate being
// number of items remaining, and the second being remaining weight. Again, this interpretation is
// irrelevant and not needed since we just need buffer of dimension n*total_knapsack_space.

// In this form, it only works if weights are integer numbers, since they are used in buffer indexing.

// If weights are not integer numbers, then one possibility is to map them to integers, thus accepting
// small error but gaining speed.
// Another problem is that weights can be very large numbers or be scattered (have large variance).
// In that case, we would need a mapping that would move them to lower value in the case they are
// large but nicely grouped, or a mapping that would essentially be a non-cryptographic hash in the
// case of large variance.

// Compared to the most basic recursive algorithm that is exponential in time, this one is quadratic,
// which is a major difference.
// The price we pay for that is that we need space for the table that has quadratic complexity.
// Of course, in specific situations, we could also experiment with different table sizes in
// range [0, (n*total_knapsack_space)], thus taking more granular tradeoff between space and time
// complexity.
// Additionally, like the basic recursive solution, we need to take into account the default stack
// size if the input is large.

Item solve(Item* items, u32 n, u32 knapsack_space, Item* buffer, u32 total_knapsack_space) {
	// If no more space or no more items.
	if(knapsack_space <= 0 || n == 0) {
		return (Item){0, 0};
	}

	// Just an index calculation since we represent buffer as an array instead of matrix.
	u32 current_item_index = ((n-1) * (total_knapsack_space + 1) + knapsack_space);

	// Buffer items values are initialized to -1, so if the value is not -1, then we have already
	// calculated that entry and we can just return it without again recursively calling function.
	if(buffer[current_item_index].value >= 0) {
		return buffer[current_item_index];
	}
	
	// Exclude currently last item if its weight exceeds knapsack space and recurse further.
	if((u32)items[n-1].weight > knapsack_space) {
		buffer[current_item_index] = solve(items, n-1, knapsack_space, buffer, total_knapsack_space);
		return buffer[current_item_index];
	}

	// Go through cases where we exclude currently last item and where we include it.
	// If we include it, then we count its value and weight and reduce current knapsack space.
	
	// We pick the path that gives maximum value for the current step (local computational maximum).
	Item value_excluded = solve(items, n-1, knapsack_space, buffer, total_knapsack_space);
	Item value_included = solve(items, n-1, knapsack_space - (u32)items[n-1].weight, buffer, total_knapsack_space);

	value_included.weight += (u32)items[n-1].weight;
	value_included.value  += (u32)items[n-1].value;

	if(value_excluded.value > value_included.value) {
		buffer[current_item_index] = value_excluded;
	}
	else {
		buffer[current_item_index] = value_included;
	}

	return buffer[current_item_index];
}

Item recursive_solution_with_2D_buffer_and_integer_weights(Item* items, u32 n, u32 knapsack_space) {
	// knapsack_space + 1 because we need weight range [0, knapsack_space].
	// n used instead of (n+1) because we handle n = 0 case explicitly with if.
	u32 buffer_size = n*(knapsack_space + 1);
	Item* buffer = malloc(buffer_size*sizeof(Item));

	for(u32 i = 0; i < buffer_size; ++i) {
		buffer[i].value = -1;
	}
	
	Item result = solve(items, n, knapsack_space, buffer, knapsack_space);
	
	free(buffer);
	return result;
}

// =========================================================================================================

int main(void) {
	u32 seed = 1234;
	u32 n = 15;
	f32 knapsack_space = 155;
	
	Item* items = generate_random_items(n, seed, 100, 100);
	if(items) {
		//print_items(items, n);
		//Item result = recursive_solution(items, n, knapsack_space);
		Item result = recursive_solution_with_2D_buffer_and_integer_weights(items, n, (u32)knapsack_space);
		printf("Maximum weight: %lf\n", result.weight);
		printf("Maximum value : %lf\n", result.value);
	}
	
	return 0;
}
