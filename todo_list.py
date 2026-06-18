todo_list = []

def add_task(task):
    todo_list.append(task)

def show_tasks():
    print("오늘의 할 일 목록")
    for index, task in enumerate(todo_list, start=1):
        print(index, ".", task)

add_task("GitHub 저장소 만들기")
add_task("소스코드 업로드하기")
add_task("커밋 기록 확인하기")

show_tasks()
