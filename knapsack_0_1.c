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
			items[i].weight = (u32)((f32)rand()/(f32)RAND_MAX * max_item_weight);
			items[i].value = (u32)((f32)rand()/(f32)RAND_MAX * max_item_value);
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
// Time complexity  :
// Space complexity :
//                   
// =========================================================================================================

Item solve(Item* items, u32 n, u32 knapsack_space, Item* buffer, u32 total_knapsack_space) {
	// If no more space or no more items.
	if(knapsack_space <= 0 || n == 0) {
		return (Item){0, 0};
	}
	
	u32 current_item_index = ((n-1) * total_knapsack_space + (knapsack_space - 1));
	
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
	u32 buffer_size = n*knapsack_space;
	Item* buffer = malloc(buffer_size*sizeof(Item));

	for(u32 i = 0; i < buffer_size; ++i) {
		buffer[i].value = -1;
	}
	
	Item result = solve(items, n, knapsack_space, buffer, knapsack_space);
	
	free(buffer);
	return result;
}

int main(void) {
	u32 seed = 1234;
	u32 n = 10;
	f32 knapsack_space = 150;
	
	Item* items = generate_random_items(n, seed, 100, 100);
	if(items) {
		print_items(items, n);
		//Item result = recursive_solution(items, n, knapsack_space);
		Item result = recursive_solution_with_2D_buffer_and_integer_weights(items, n, (u32)knapsack_space);
		printf("Maximum weight: %lf\n", result.weight);
		printf("Maximum value : %lf\n", result.value);
	}
	
	return 0;
}
