def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    
    # Search actions
    paths = util.Queue()
    # Record states (x, y) that have been put into stack
    visited = []
    # bredth first search queue
    queue = util.Queue()
    
    # Push the first item's successors to the stack
    visited.append(problem.getStartState())
    for x in problem.getSuccessors(problem.getStartState()):
        queue.push(x)
        visited.append(x[0])
        paths.push([x[1]])

    while not queue.isEmpty():
        current = queue.pop()
        curPath = paths.pop()
        
        # If found the goal state, break the loop
        if problem.isGoalState(current[0]):
            return curPath

        # Get current state's successors
        successors = problem.getSuccessors(current[0])

        for x in successors:
            if x[0] not in visited:
                visited.append(x[0])
                queue.push(x)
                paths.push(curPath + [x[1]])
    
    # When reach here, it means that we cannot found the goal
    return []


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    ----
    Start: (34, 16)
    Is the start a goal? False
    Start's successors: [((34, 15), 'South', 1), ((33, 16), 'West', 1)]
    Path found with total cost of 999999 in 0.0 seconds
    """
    "*** YOUR CODE HERE ***"
    # Search actions
    paths = []
    # Record states (x, y) that have been put into stack
    visited = []
    # Depth first search stack
    stack = util.Stack()
    
    # Push the first item's successors to the stack
    visited.append(problem.getStartState())
    successors = problem.getSuccessors(problem.getStartState())

    # Add explorer random searching order
    priority = random.choice(["South", "West"])
    prioritySuccessors = []
    for x in successors:
        if x[1] == priority:
            prioritySuccessors.append(x)
        else:
            prioritySuccessors.insert(0, x)

    for x in prioritySuccessors:
        stack.push(x)
        visited.append(x[0])
        paths.append([])

    while not stack.isEmpty():
        current = stack.pop()
        paths[-1].append(current[1])
        # Found the goal state, break the loop
        if problem.isGoalState(current[0]):
            break

        # Get current state's successors
        successors = problem.getSuccessors(current[0])

        # Default search priority is North, South, East, West
        # Add explorer random searching order
        priority = random.choice(["South", "West", "East", "North"])

        # Check whether this state has successors that are not in "visited" list.
        newBranch = 0
        prioritySuccessors = []
        for x in successors:
            if x[0] not in visited:
                if x[1] == priority or x[1]:
                    prioritySuccessors.append(x)
                else:
                    prioritySuccessors.insert(0, x)
                newBranch += 1

        if newBranch == 0:
            # Didn't find the goal state and has no successor. 
            # This is a wrong path and should be discarded.
            while paths[-1] != []:
                paths.pop(-1)

        for x in prioritySuccessors:
            if x[0] not in visited:
                stack.push(x)
                visited.append(x[0])
                if newBranch > 1:
                    # When newBranch > 1, it has at least two branches
                    paths.append([])

    # Combine search paths
    searchAction = []
    for path in paths:
        searchAction += path
    return searchAction

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    # Search paths so far
    paths = util.PriorityQueue()
    # Record states (x, y) that have been put into stack
    visited = []
    # Stored actions for searching
    storedAct = util.PriorityQueue()

    # Check current state is the goal state or not.
    if problem.isGoalState(problem.getStartState()):
        return []

    # Push the first item's successors to the stack
    visited.append(problem.getStartState())
    for x in problem.getSuccessors(problem.getStartState()):
        storedAct.push(x, x[2])
        visited.append(x[0])
        paths.push([x[1]], x[2])

    while not storedAct.isEmpty():
        current = storedAct.pop()
        curPath = paths.pop()

        # Found the goal state, break the loop
        if problem.isGoalState(current[0]):
            return curPath

        # Get current state's successors
        successors = problem.getSuccessors(current[0])

        for x in successors:
            if x[0] not in visited:
                visited.append(x[0])
                storedAct.push(x, x[2])
                paths.push(curPath + [x[1]], x[2])

    # When reach here, it means that we cannot found the goal
    return []    