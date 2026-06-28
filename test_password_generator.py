import unittest
import os
import tempfile
import json
from cryptography.fernet import Fernet
import numpy as np
from password_generator import (
    generate_password,
    extract_features,
    entropy,
    has_sequence,
    repeat_ratio,
    count_classes,
    evaluate_password,
    get_cipher,
    save_password_record,
    load_history,
    train_and_save_model,
    load_model
)

class TestPasswordGenerator(unittest.TestCase):

    #  1. Тесты генерации  

    def test_generate_length(self):
        """Проверка, что длина пароля соответствует заданной"""
        for length in [6, 12, 30]:
            pwd = generate_password(length=length, use_digits=True, use_special=True,
                                    use_upper=True, use_lower=True)
            self.assertEqual(len(pwd), length)

    def test_generate_only_digits(self):
        """Проверка, что при выборе только цифр пароль состоит только из цифр"""
        pwd = generate_password(length=10, use_digits=True, use_special=False,
                                use_upper=False, use_lower=False)
        self.assertTrue(pwd.isdigit())

    def test_generate_avoid_ambiguous(self):
        """Проверка исключения неоднозначных символов (il1Lo0O)"""
        pwd = generate_password(length=20, avoid_ambiguous=True,
                                use_digits=True, use_special=True,
                                use_upper=True, use_lower=True)
        ambiguous = 'il1Lo0O'
        for ch in pwd:
            self.assertNotIn(ch, ambiguous)

    def test_generate_no_chars_error(self):
        """Проверка, что при отключении всех типов символов выбрасывается исключение"""
        with self.assertRaises(ValueError):
            generate_password(length=8, use_digits=False, use_special=False,
                              use_upper=False, use_lower=False)

    #  2. Тесты признаков  

    def test_entropy_calculation(self):
        """Проверка расчёта энтропии для известных строк"""
        self.assertAlmostEqual(entropy("aaaaaa"), 0.0, places=5)
        self.assertAlmostEqual(entropy("ababab"), 1.0, places=5)
        self.assertGreater(entropy("qwertyuiop"), entropy("1234567890"))

    def test_has_sequence(self):
        """Проверка обнаружения клавиатурных последовательностей"""
        self.assertEqual(has_sequence("qwerty123"), 1)
        self.assertEqual(has_sequence("abc123"), 1)
        self.assertEqual(has_sequence("йцукен"), 1)
        self.assertEqual(has_sequence("random"), 0)

    def test_repeat_ratio(self):
        """Проверка доли уникальных символов"""
        self.assertAlmostEqual(repeat_ratio("aaaaaa"), 1/6, places=5)
        self.assertEqual(repeat_ratio("abcdef"), 1.0)
        self.assertEqual(repeat_ratio(""), 0.0)

    def test_extract_features_shape(self):
        """Проверка, что извлекается вектор из 9 признаков"""
        features = extract_features("Test123!")
        self.assertEqual(features.shape, (9,))
        self.assertEqual(features.dtype, np.float64)

    #  3. Тесты оценки модели  

    def test_evaluate_weak_password(self):
        """Проверка, что заведомо слабый пароль получает низкую оценку"""
        model = load_model()
        prob, strength = evaluate_password("123456", model)
        self.assertLess(prob, 0.3)
        self.assertIn(strength, ["Очень слабый", "Слабый"])

    def test_evaluate_strong_password(self):
        """Проверка, что заведомо сильный пароль получает высокую оценку"""
        model = load_model()
        strong_pwd = "aB3!xZ9pQw&"
        prob, strength = evaluate_password(strong_pwd, model)
        self.assertGreater(prob, 0.7)
        self.assertEqual(strength, "Очень надёжный")

    #  4. Тесты шифрования  

    def test_encryption_decryption(self):
        """Проверка, что пароль шифруется и дешифруется корректно"""
        original = "SecretPass123!"
        cipher = get_cipher()
        encrypted = cipher.encrypt(original.encode())
        decrypted = cipher.decrypt(encrypted).decode()
        self.assertEqual(original, decrypted)

    def test_save_and_load_history(self):
        """Проверка сохранения и загрузки истории с шифрованием"""
        fd, temp_path = tempfile.mkstemp(suffix='.json')
        os.close(fd)
        save_password_record("TestPwd", "Средней надёжности", 0.65, filename=temp_path)
        history = load_history(temp_path)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]['password'], "TestPwd")
        self.assertEqual(history[0]['strength'], "Средней надёжности")
        self.assertAlmostEqual(history[0]['probability'], 0.65, places=2)
        os.remove(temp_path)
        os.remove('secret.key')


if __name__ == '__main__':
    unittest.main()