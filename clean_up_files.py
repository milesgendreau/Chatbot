import os

def clean_up_file(episode_num, file_name):
    file = open(file_name,'r',encoding='utf-8')

    header = file.readline()
    lines = file.readlines()

    title = ''.join(lines[0].split('","')[1].split(':')[2:])
    title = title.strip('"').replace(',','')
    print(title)
    
    for i, line in enumerate(lines):
        if line[0] != '"':
            lines[i-1] = lines[i-1].replace('\n', ' ')
            lines[i-1] += line
            lines.pop(i)

    lines = sorted(lines)

    for i, line in enumerate(lines):
        l = line.split('","')[2:]
        if len(l) > 0 and l[0] == '':
            l[0] = 'n/a'
        l = ','.join(l)
        lines[i] = l.replace('"','')

    file.close()

    file = open('avatar-scripts/(' + str(episode_num) + ')' + title + '.txt', 'w', encoding='utf-8')
    file.write('speaker,line\n')
    file.writelines(lines)
    file.close()


for i in range(2, 61):
    clean_up_file(i, 'temp/avatar (' + str(i-1) + ').csv')
