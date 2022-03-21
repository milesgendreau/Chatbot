import os

files = os.listdir('avatar-scripts')
target_characters = ['Sokka', 'Toph', 'Iroh']

def gather_data(file, data_file):
    header = file.readline()
    lines = file.readlines()

    prev_speaker = ''
    prev_line = ''
    for elem in lines:
        sep = elem.split(',', 1)
        if len(sep) < 2:
            sep = ['', '']
        speaker, line = sep[0], sep[1]
        line = remove_brackets(line)
        
        if speaker != 'n/a':
            if speaker != prev_speaker and prev_speaker != '' and speaker in target_characters:
                data_file.write(prev_line + ';' + line + '\n')

            prev_speaker = speaker
            prev_line = line

def remove_brackets(line):
    new_str = ''
    remove = False
    for char in line:
        if char == '[':
            remove = True

        if not remove:
            new_str += char

        if char == ']':
            remove = False

    return new_str.strip()

data_file = open('training_data.txt', 'w', encoding='utf-8')
for filename in files:
    file = open('avatar-scripts/' + filename, 'r', encoding='utf-8')
    gather_data(file, data_file)

