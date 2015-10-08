Artificial Intelligence Homework #2-Game Playing
Group Member: Li-Yi Lin / llin34@jhu.edu



Please use the following commands to run the searching algorithms:
Question 1-DFS============================================================
python pacman.py -l tinyMaze -p SearchAgent --frameTime 0
python pacman.py -l mediumMaze -p SearchAgent --frameTime 0
python pacman.py -l bigMaze -z .5 -p SearchAgent --frameTime 0

Question 2-BFS============================================================
python pacman.py -l tinyMaze -p SearchAgent -a fn=bfs --frameTime 0
python pacman.py -l mediumMaze -p SearchAgent -a fn=bfs --frameTime 0
python pacman.py -l bigMaze -p SearchAgent -a fn=bfs -z .5 --frameTime 0

Grad Student-IDS==========================================================
python pacman.py -l tinyMaze -p SearchAgent -a fn=ids --frameTime 0
python pacman.py -l mediumMaze -p SearchAgent -a fn=ids --frameTime 0
python pacman.py -l bigMaze -p SearchAgent -a fn=ids -z .5 --frameTime 0

Question 3-UCS============================================================
python pacman.py -l mediumMaze -p SearchAgent -a fn=ucs --frameTime 0
python pacman.py -l mediumDottedMaze -p StayEastSearchAgent --frameTime 0
python pacman.py -l mediumScaryMaze -p StayWestSearchAgent --frameTime 0

Grad Student-UCS new cost func============================================
python pacman.py -l mediumMaze -p NewUCSCostSearchAgent --frameTime 0
python pacman.py -l mediumDottedMaze -p NewUCSCostSearchAgent --frameTime 0
python pacman.py -l mediumScaryMaze -p NewUCSCostSearchAgent --frameTime 0

Question 4-astar==========================================================
python pacman.py -l tinyMaze -z .5 -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic --frameTime 0
python pacman.py -l mediumMaze -z .5 -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic --frameTime 0
python pacman.py -l bigMaze -z .5 -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic --frameTime 0

Question 5-Corner-BFS=====================================================
python pacman.py -l tinyCorners -p SearchAgent -a fn=bfs,prob=CornersProblem --frameTime 0
python pacman.py -l mediumCorners -p SearchAgent -a fn=bfs,prob=CornersProblem --frameTime 0

Question 6-Corner-astar===================================================
python pacman.py -l tinyCorners -p AStarCornersAgent -z 0.5 --frameTime 0
python pacman.py -l mediumCorners -p AStarCornersAgent -z 0.5 --frameTime 0

Question 7-Food-astar=====================================================
python pacman.py -l testSearch -p SearchAgent -a fn=astar,prob=FoodSearchProblem --frameTime 0
python pacman.py -l trickySearch -p SearchAgent -a fn=astar,prob=FoodSearchProblem --frameTime 0

Grad Student-Food-astar-food heuristic====================================
python pacman.py -l testSearch -p AStarFoodSearchAgent --frameTime 0
python pacman.py -l trickySearch -p AStarFoodSearchAgent --frameTime 0



