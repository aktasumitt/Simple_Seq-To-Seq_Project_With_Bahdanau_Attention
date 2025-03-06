# Seq2Seq Model with Bahdanau Attention

Bu proje, en fazla 8 uzunlukta giriş dizisi alıp dizinin tam tersini döndüren basit bir seq-to-seq modelidir. Model, **Bahdanau Attention** ve **LSTM** katmanları kullanılarak eğitilmiştir. Modelin tahmin yapmasını sağlayan bir **Flask uygulaması** bulunmaktadır.

## Proje Yapısı

- **src**: Source code.
- **app.py**: Flask uygulaması, model tahminlerini almak için kullanılır.
- **requirements.txt**: Gerekli bağımlılıkları içerir.
- **README.md**: Proje hakkında bilgi sağlar.

## Kurulum

1. **Gerekli bağımlılıkları yükleyin**

   ```sh
   pip install -r requirements.txt
   ```

2. **Flask uygulamasını çalıştırın**

   ```sh
   python app.py
   ```

3. **API'yi test edin**
   Flask uygulaması çalıştıktan sonra, "localhost:5000" üzerinde calısacaktır:


## Model Hakkında

Bu model, **encoder-decoder** yapısı kullanarak seq-to-seq (sequence-to-sequence) dönüşümleri gerçekleştirmektedir. **Bahdanau Attention** mekanizması, modelin belirli giriş öğelerine odaklanmasını sağlar.

Eğitim sürecinde, modelin **maksimum 8 uzunluğunda dizileri** ters çevirmesi amaçlanmıştır.

## API Kullanımı

- **Girdi**: `{"sequence": "1 2 3 4 5 6 7 8"}`
- **Çıktı**: `{"output": "8 7 6 5 4 3 2 1"}`



