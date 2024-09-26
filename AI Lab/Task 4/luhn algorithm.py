#Luhn Algorithms
class Luhn:
    def __init__(self,card_number):
        self.card_number= card_number
    def remove(self):
            self.e =self.card_number.pop()
            print("Digit is",self.e)
    def reverse(self):
            self.card_number.reverse()
            print("Reverse Digit is",self.card_number)
    def even_indexing(self):
            for i in range(len(self.card_number)):
                if i%2==0:
                    self.card_number[i]*=2
                    if self.card_number[i]>1:
                        self.card_number[i]-=1
            print("Even-index digits is ",self.card_number)
    def check_valid(self):
            total = sum(self.card_number)+self.e
            if total%10==0:
                print("Card is Valid")
            else:
                print("Card is Invalid")   

card=Luhn([3,3,4,0,1,0,4,1,2,0,3,3,1])
card.remove()
card.reverse()
card.even_indexing()
card.check_valid()