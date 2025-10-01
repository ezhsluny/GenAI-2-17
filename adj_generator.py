import re
from generator import generate_text, initialize_generator, chat_prompt
import argparse


def count_mentions(text, word):
    """Считает количество упоминаний слова в тексте (регистронезависимо)."""
    return len(re.findall(rf'\b{re.escape(word)}\b', text, flags=re.IGNORECASE))

def generate_and_check(prompt, model):
    """
    Генерирует историю и проверяет количество упоминаний слова "Волк".
    Args:
        prompt: Текстовый запрос.
        model: Название модели.
    Returns:
        Сгенерированная история и количество упоминаний.
    """
    generator = initialize_generator(model)
    if not generator:
        raise RuntimeError("Не удалось инициализировать генератор!")

    messages = chat_prompt(prompt)
    story = generate_text(
        generator, messages,
        max_new_tokens=300,      # Ограничение длины
        do_sample=True,          # Случайность для креативности
        temperature=0.6,         # Температура генерации (хаотичность)
        top_p=0.9,               # Контроль разнообразия
        repetition_penalty=1.3   # Контроль повторов слов и фраз 
    )

    mentions = count_mentions(story, "Волк")
    return story, mentions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Генератор истории по заданному промпту")
    parser.add_argument("--prompt", type=str, help="Текстовый запрос для генерации истории")
    args = parser.parse_args()

    if args.prompt is None:
        prompt = 'Напиши короткую историю про лесного Волка, который живёт в чаще.\
              В истории должно быть ровно 3 упоминания слова "Волк":\
              1. Когда его впервые увидели.\
              2. Когда он выл на луну ночью.\
              3. Когда он исчез в лесу.'
    else:
        prompt = args.prompt

    model = "Qwen/Qwen2-1.5B-Instruct"
    story, mentions = generate_and_check(prompt, model)

    print("=== СГЕНЕРИРОВАННАЯ ИСТОРИЯ ===")
    print(story)
    print(f"\nКоличество упоминаний слова 'Волк': {mentions}")

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