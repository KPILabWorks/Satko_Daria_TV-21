import random
# Використовуючи множини, знайти всі унікальні елементи в кількох списках.

def random_list(size, start=1, end=20): # Робимо список з діапазоном
    return [random.randint(start, end) for _ in range(size)]

def unique_elements(*lists):# Шукаємо унікальні елементи
    return set().union(*lists)

list1 = random_list(10) # Створюємо списки
list2 = random_list(10)
list3 = random_list(10)

unique_set = unique_elements(list1, list2, list3) # Шукаємо унікальні елементи

print("Список 1:", list1) # Виводимо списки
print("Список 2:", list2)
print("Список 3:", list3)
print("Унікальні елементи:", unique_set)
