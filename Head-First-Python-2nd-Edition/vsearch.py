def search4Vowels(phrase: str) -> set:
    """Return any vowels found in a supplied phrase."""
    vowels = set('aeiou')
    found = vowels.intersection(set(word))
    for vowel in found:
        print(vowel)

    return found


def search4letters(phrase: str, letters: str='aeiou') -> set:
    """Return a set of 'letters' found in 'phrase'."""
    return set(letters).intersection(set(phrase))
