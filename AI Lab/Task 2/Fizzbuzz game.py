# Fizz Buzz Game
import random
class FizzBuzz:
    def __init__(self, n):
        self.n = n
    def game_logic(self, num):
            if num % 3 == 0 and num % 5 == 0:
                return 'fizz_buzz'
            elif num % 3 == 0:
                return 'fizz'
            elif num % 5 == 0:
                return 'buzz'
            else:
                return num
    def play(self):
        print("Ready to play Game :")
        while True:
            num = random.randint(1, self.n)
            print("Number is:", num)
            player_answer = input("Your answer is ").strip()
            check_answer = self.game_logic(num)
            if player_answer == check_answer:
                print("Your answer correct. Continue the game")
            else:
                print(f"Wrong answer. The correct answer was {check_answer}.")
                print("Game Over")
                break
obj = FizzBuzz(100)
obj.play()

