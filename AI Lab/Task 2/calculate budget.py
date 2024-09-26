# Question02
movies = [
    ("Megamind", 20000000),
    ("Avatar", 9000000),
    ("12 Fail", 45000),
    ("Pirates of the Caribbean: No man Dead", 379000000),
    ("Man of Steel", 323000000),
    ("Inside Out", 345000),
    ("Extraction", 150000000)
]
new_movies = int(input("How many new movies wants to add:"))
for _ in range(new_movies):
    name = input("Enter the new movie name: ")
    budget = int(input("Enter the movie budget: "))
    new_movie = (name, budget)
    movies.append(new_movie)

higher_budget_movies = []
total_budget = 0
for movie in movies:
    total_budget = total_budget + movie[1]
average_budget = int(total_budget // len (movies)) 
for movie in movies:
    if movie[1] > average_budget:
        higher_budget_movies.append(movie)
        over_budget_cost = movie[1] - average_budget
        print(f"{movie[0]} cost ${movie[1]:,} ${over_budget_cost:,}over budget.")
    print(f"There are {len(higher_budget_movies)} movies with over budget cost..")