import random

ans_max=10
ans_min=0
ans_cha=10

ans=random.randint(ans_min,ans_max)
      
for k in range(ans_cha):
 x=int(input(f'{ans_min}から{ans_max}の値を入れてください'))
 if x>ans:
  print('もっと大きいです')
 elif x<ans:
  print('もっと小さいです')
 else:
  print
  break
print('おしまい')