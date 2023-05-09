# Aligning-contextual-vector-spaces
Aligning contextual vector spaces between independent neural translation systems

Abstract:
Numerous pre-trained machine translation models are available for translating between
different languages. However, these models are limited to a fixed set of languages they
were trained for. When there is no translation model available for a specific language
pair, we need to translate to one or more intermediate languages, which can result in
reduced translation quality. We investigate the possibility of combining two translation
models by aligning the vector spaces between them using a simple regressor. We explore
the effectiveness of various regression methods for achieving this alignment and evaluate
their performance. We show that combining two different translation models is possible,
although doing so leads to a decrease in translation quality
