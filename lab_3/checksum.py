import json
import hashlib
from typing import List

"""
В этом модуле обитают функции, необходимые для автоматизированной проверки результатов ваших трудов.
"""
"email";"http_status_message";"inn";"passport";"ip_v4";"latitude";"hex_color";"isbn";"uuid";"time"

patterns = {
    'email': r'^(?!.*\.\.)(?!.*\.$)(?!^\.)[a-zA-Z0-9]+(?:\.[a-zA-Z0-9]+)*@(?!.*\.\.)(?!.*\.$)(?!^\.)[a-zA-Z0-9]+(?:\.[a-zA-Z0-9]+)+\.[a-zA-Z]{2,}$',
    'http_status_message': r'^\d{3} [A-Za-z]+( [A-Za-z]+)*$',
    'inn': r'^\d{12}$',
    'passport': r'^\d{2} \d{2} \d{6}$',
    'ip_v4': r'^((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'

    # 'latitude': 
    # 'hex_color':
    # 'isbn':
    # 'uuid':
    # 'time':
}


def calculate_checksum(row_numbers: List[int]) -> str:
    """
    Вычисляет md5 хеш от списка целочисленных значений.

    ВНИМАНИЕ, ВАЖНО! Чтобы сумма получилась корректной, считать, что первая строка с данными csv-файла имеет номер 0
    Другими словами: В исходном csv 1я строка - заголовки столбцов, 2я и остальные - данные.
    Соответственно, считаем что у 2 строки файла номер 0, у 3й - номер 1 и так далее.

    Надеюсь, я расписал это максимально подробно.
    Хотя что-то мне подсказывает, что обязательно найдется человек, у которого с этим возникнут проблемы.
    Которому я отвечу, что все написано в докстринге ¯\_(ツ)_/¯

    :param row_numbers: список целочисленных номеров строк csv-файла, на которых были найдены ошибки валидации
    :return: md5 хеш для проверки через github action
    """
    row_numbers.sort()
    return hashlib.md5(json.dumps(row_numbers).encode('utf-8')).hexdigest()


def serialize_result(variant: int, checksum: str) -> None:
    """
    Метод для сериализации результатов лабораторной пишите сами.
    Вам нужно заполнить данными - номером варианта и контрольной суммой - файл, лежащий в папке с лабораторной.
    Файл называется, очевидно, result.json.

    ВНИМАНИЕ, ВАЖНО! На json натравлен github action, который проверяет корректность выполнения лабораторной.
    Так что не перемещайте, не переименовывайте и не изменяйте его структуру, если планируете успешно сдать лабу.

    :param variant: номер вашего варианта
    :param checksum: контрольная сумма, вычисленная через calculate_checksum()
    """
    pass


if __name__ == "__main__":
    print(calculate_checksum([1, 2, 3]))
    print(calculate_checksum([3, 2, 1]))
