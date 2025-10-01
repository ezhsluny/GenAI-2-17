# Установите зависимости: pip install transformers torch
# Рекомендуемая версия transformers >= 4.34 (для поддержки chat-формата и device_map)
from transformers import pipeline
import torch
import sys
import argparse
import importlib

# Пытаемся импортировать transformers динамически
try:
    transformers = importlib.import_module("transformers")
    required_version = "4.34"
    current_version = transformers.__version__

    # Сравниваем версии
    if tuple(map(int, current_version.split('.')[:2])) < tuple(map(int, required_version.split('.'))):
        print(f"Требуется transformers>={required_version}, у вас {current_version}")
        print("Обновите: pip install --upgrade transformers")
        sys.exit(1)

except ImportError:
    print("Библиотека 'transformers' не установлена!")
    print("Установите: pip install transformers torch")
    sys.exit(1)


def initialize_generator(model):
    """Создание pipeline генерации текста с указанной моделью
    Args: model (str). Название модели.
    Returns: Pipeline or None."""
    if not isinstance(model, str) or not model.strip():
        raise ValueError("'model' должен быть непустой строкой!")

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    try:
        generator = pipeline(
            "text-generation",
            model=model,
            device_map="auto",  # Автоматически выберет CPU или GPU
            dtype=dtype
        )
        return generator
    except Exception as e:
        print(f"Не удалось загрузить модель '{model}': {e}")
        print("Возможные причины: нет интернета, мало памяти, ошибка совместимости.")
        return None


def chat_prompt(prompt):
    """Форматирование текстового запроса в формат чата для инструктивных моделей
    Args: prompt (str). Текстовый запрос пользователя.
    Returns: messages (list[dict]). Список с одним сообщением от пользователя"""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("'prompt' должен быть непустой строкой!")

    # Форматирование запроса в чат-стиле (это важно для Qwen2)
    messages = [
        {"role": "user", "content": prompt}
    ]
    return messages


# Генерируем историю
def generate_text(generator, messages, max_new_tokens, do_sample, temperature, top_p, repetition_penalty):
    """Генерация текста с заданными параметрами.
    Args:
        generator (pipeline): Инициализированный pipeline.
        messages (list[dict]): Сообщения в формате чата.
        max_new_tokens (int): Макс. количество новых токенов.
        do_sample (bool): Использовать ли сэмплирование.
        temperature (float): Температура генерации.
        top_p (float): Параметр ядра для разнообразия.
        repetition_penalty (float): Штраф за повторы.
    Returns: Сгенерированный текст."""
    if not isinstance(max_new_tokens, int) or max_new_tokens <= 0:
        raise ValueError("'max_new_tokens' должен быть положительным целым числом!")
    if not isinstance(do_sample, bool):
        raise ValueError("'do_sample' должен быть булевым значением!")
    if not isinstance(temperature, (int, float)) or temperature < 0:
        raise ValueError("'temperature' должен быть числом >= 0!")
    if not isinstance(top_p, (int, float)) or not (0 < top_p <= 1.0):
        raise ValueError("'top_p' должен быть числом в диапазоне (0, 1]!")
    if not isinstance(repetition_penalty, (int, float)) or repetition_penalty <= 0:
        raise ValueError("'repetition_penalty' должен быть числом > 0!")
    if not isinstance(messages, list) or len(messages) == 0 or not isinstance(messages[0], dict):
        raise ValueError("'messages' должен быть списком словарей с ключами 'role' и 'content'!")

    result = generator(
        messages,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty
    )

    if not result or not isinstance(result, list) or len(result) == 0:
        raise ValueError("Генератор вернул пустой результат!")
    if 'generated_text' not in result[0]:
        raise ValueError("В результате отсутствует ключ 'generated_text'!")
    generated_text = result[0]['generated_text']
    if not isinstance(generated_text, list) or len(generated_text) == 0:
        raise ValueError("'generated_text' пуст или не является списком!")
    last_message = generated_text[-1]
    if not isinstance(last_message, dict) or 'content' not in last_message:
        raise ValueError("Последнее сообщение не содержит ключ 'content'!")

    generated_story = result[0]['generated_text'][-1]['content']
    return generated_story


def main():
    """Это основная функция. Она инициализирует модель, генерирует историю, сохраняет в файл"""

    parser = argparse.ArgumentParser(description="Генератор истории по заданному промпту")
    parser.add_argument("--prompt", type=str, help="Текстовый запрос для генерации истории")
    args = parser.parse_args()

    if args.prompt is None:
        prompt = "Напиши историю, используя слова: лес, волк, дом"
    else:
        prompt = args.prompt

    model = "Qwen/Qwen2-1.5B-Instruct"
    max_new_tokens = 400        # Ограничение длины
    do_sample = True            # Случайность для креативности
    temperature = 0.7           # Баланс: 0.7 - не слишком хаотично, не слишком скучно
    top_p = 0.9                 # Контроль разнообразия
    repetition_penalty = 1.2    # Контроль повторов слов и фраз

    generator = initialize_generator(model)
    messages = chat_prompt(prompt)

    story = generate_text(generator, messages, max_new_tokens, do_sample, temperature, top_p, repetition_penalty)
    # Вывод сгенерированного текста без промпта

    print("=== СГЕНЕРИРОВАННАЯ ИСТОРИЯ ===")
    print(story)

    output_file = "output.txt"
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(story)
        print(f"\nИстория успешно сохранена в файл: {output_file}")
    except PermissionError:
        print(f"\nОшибка: Нет прав на запись в файл '{output_file}'.")
    except OSError as e:
        print(f"\nОшибка при записи в файл: {e}")
    except Exception as e:
        print(f"\nНеизвестная ошибка при сохранении: {e}")

if __name__ == "__main__":
    main()