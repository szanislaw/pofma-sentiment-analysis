from typing import List
# nums: List[int]
# target: int

def sort_and_search(nums: List[int], target: int) -> bool:
    nums.sort()
    
    left = 0 
    right = len(nums) 
    
    
    while left <= right:
        
        mid = (left+right) // 2 
        
        if nums[mid] == target:
            return True
        
        elif nums[mid] <= target:
            left = mid + 1
            return True
            
        else:
            right = mid 
            
        return False
        # print(mid)

        

print(sort_and_search([4, 2, 3, 1, 5], 3))  # Output: True  
print(sort_and_search([10, 7, 5, 3, 8], 6))  # Output: False  
print(sort_and_search([1, 2, 3, 4, 5, 5], 5))  # Output: True  

#return a bool