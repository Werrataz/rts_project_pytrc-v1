import time
import csv
from pynput import keyboard

# Variables to store data
keystrokes = []
start_time = None
enrol_errors = 0

# CSV setup
keystroke_file = open('keystroke_data.csv', mode='w', newline='')
keystroke_writer = csv.writer(keystroke_file)
keystroke_writer.writerow(['event_type', 'key_code', 'timestamp'])

feature_file = open('features_data.csv', mode='w', newline='')
feature_writer = csv.writer(feature_file)
feature_writer.writerow(['press_press', 'release_release', 'press_release', 'release_press', 'vector'])

def on_press(key):
    global start_time
    if start_time is None:
        start_time = time.time()
    timestamp = time.time()
    keystrokes.append(('press', key, timestamp))
    keystroke_writer.writerow(['press', str(key), timestamp])

def on_release(key):
    timestamp = time.time()
    keystrokes.append(('release', key, timestamp))
    keystroke_writer.writerow(['release', str(key), timestamp])
    if key == keyboard.Key.enter:
        return False

def calculate_features(keystrokes):
    press_press = []
    release_release = []
    press_release = []
    release_press = []
    last_press = None
    last_release = None

    for event in keystrokes:
        if event[0] == 'press':
            if last_press is not None:
                press_press.append(event[2] - last_press)
            if last_release is not None:
                release_press.append(event[2] - last_release)
            last_press = event[2]
        elif event[0] == 'release':
            if last_release is not None:
                release_release.append(event[2] - last_release)
            if last_press is not None:
                press_release.append(event[2] - last_press)
            last_release = event[2]

    vector = press_press + release_release + press_release + release_press
    feature_writer.writerow([str(press_press), str(release_release), str(press_release), str(release_press), str(vector)])

def main():
    print("Please type your text. Press Enter to finish and save the data.")
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

    password = ''.join([str(k[1]) for k in keystrokes if k[1] != keyboard.Key.enter])
    time_to_type = time.time() - start_time
    calculate_features(keystrokes)

    print(f"Password: {password}")
    print(f"Time to type: {time_to_type} seconds")
    print(f"Number of enrol errors: {enrol_errors}")

if __name__ == "__main__":
    main()
    keystroke_file.close()
    feature_file.close()

# Trouver un moyen de récupérer les données (peut-être en les enregistrant sur un csv sur un drive plutôt qu'en local ?)
# Voir avec l'équipe qui était en charge de ça s'ils ont compris la structure des données de la bdd d'origine, et s'ils ont eu le temps de finir le logiciel de keystroke. S'ils ont pas commencé, prendre la main là dessus ?