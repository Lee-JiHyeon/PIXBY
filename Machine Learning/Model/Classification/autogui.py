import pyautogui, time
print(pyautogui.position())
while True:
    for _ in range(5):
        pyautogui.click(x=700, y=385)
    time.sleep(60)