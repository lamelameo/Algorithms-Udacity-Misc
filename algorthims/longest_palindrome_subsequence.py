import timeit


# Find longest palindrome given a sequence of letters. We will move left to right across letters and searching for
# that letter's duplicate, if any, searching right to left and taking the first match we find. We will then recurse on
# the subsequence contained between these letters and also iterate over all starting letters left to right at
# all recursion depths. We will take the max of this iteration as our return result at each depth. If we recurse on a
# 1 or 2 length sequence, then this will be our starting result to return and move up one depth level.
def palindrome(word):
    # TODO: can use a data structure to hold each letter's duplicate's indexes so we dont have to search or less search
    # may need a lot of space to store and time to create?
    # TODO: can save palindromes themselves in dict under own key, rather than just the subsequences
    # this may save time if we find that palindrome, but requires extra space
    memo = dict()
    counter = 0
    duplicates = dict()
    # TODO: making array to make searching easier
    for index, character in enumerate(word):
        if character not in duplicates:
            duplicates[character] = [index]
        else:
            duplicates[character].append(index)

    # recursive function to find longest palindrome in a given sequence
    def recurse(seq):
        nonlocal counter
        # already stored value of this subsequence
        counter += 1
        if seq in memo:
            return memo[seq]

        # otherwise must find nested duplicate letters recursively to build up palindromes
        if len(seq) > 2:
            # if our outer letters are same, we can just recurse on inner letter and add outer to return value
            if seq[0] == seq[-1]:
                pal = recurse(seq[1:-1])
                memo[seq] = seq[0] + pal + seq[0]
                return memo[seq]
            # otherwise we must search for duplicate letters inside the subsequence to find best inner palindrome
            else:
                longest = ""
                # iterate through all letters as starting points to find longest palindrome
                for index_start, letter in enumerate(seq):
                    # find duplicate by iterating backward through sequence until the letter itself
                    for index_end in range(1, len(seq) - index_start):
                        # found duplicate and thus a palindrome
                        if seq[0 - index_end] == letter:
                            # recurse on subsequence
                            if index_end == 1:
                                pal = recurse(seq[index_start:])
                            else:
                                pal = recurse(seq[index_start:1 - index_end])
                            # TODO: * had palindrome saver here
                            break
                    else:  # no duplicates, recurse will just return the letter and memoize if necessary
                        pal = recurse(letter)

                    # TODO: This saves another entry for the palindrome itself, rather than sequence
                    # TODO: may save time but requires more space, is it useful?
                    # if pal not in memo:  # store if we havent seen this yet
                    #     print('test1:', pal)
                    #     memo[pal] = pal
                    # else:
                    #     print('test2:', pal)
                    #     pass

                    # obtain max length palindrome from all letters as starting positions
                    if len(pal) > len(longest):
                        longest = pal

                memo[seq] = longest
                return longest

        # must check for length 1 or 2 words, in case we start with these
        elif len(seq) == 2:  # word 2 letters
            if seq[0] == seq[1]:  # same letter
                memo[seq] = seq
                return seq
            else:  # different letters, must return only one letter so just use the first
                memo[seq] = seq[0]
                return seq[0]

        else:  # word = 1 letter (or "" ie no sequence)
            memo[seq] = seq
            return seq

    longest_palindrome = recurse(word)
    # print("memoized:", len(memo))
    # print("recursions:", counter)
    return longest_palindrome


# print("Longest palindrome:", palindrome("aaaaaaaaaaaaaaaaaaataaaaaaaaaaaaaaaaaaaaa"), "\n")


# more simple method to get longest palindrome subsequence
def palindrome2(word):
    # store results as we determine them
    store = dict()
    counter = 0

    def recurse(seq):
        nonlocal counter
        counter += 1
        # print("sub sequence:", seq)
        # memoized already
        if seq in store:
            return store[seq]

        # length 1 subsequence
        if len(seq) == 1:
            store[seq] = seq
            return seq

        # length 2 subsequence
        elif len(seq) == 2:
            if seq[0] == seq[1]:
                store[seq] = seq
                return seq
            else:
                store[seq] = seq[0]
                return seq[0]

        # length > 2 subsequence
        else:
            # outer letters are a palindrome, must recurse and then add the result
            if seq[0] == seq[-1]:
                inner = recurse(seq[1:-1])
                store[seq] = seq[0] + inner + seq[0]
                return store[seq]
            # outer letters are different, recurse by incrementing inwards one letter on either side and take best
            else:
                left = recurse(seq[:-1])
                right = recurse(seq[1:])
                best = max(left, right, key=lambda k: len(k))
                store[seq] = best
                return best

    # recurse on word to get palindrome
    pal = recurse(word)
    # print('memoized:', len(store))
    # print('recursions:', counter)
    return pal


def test_palindrome():

    # print("Longset palindrome:", palindrome2('aaaaaaaaaaaaaaaaaaaataaaaaaaaaaaaaaaaaaaa'))
    # abcdefghijklmnopqrstuvwxyzyxwvutrqponmlkjihgfedcba aaaaaaaaaaaaaaaaaaataaaaaaaaaaaaaaaaaaaaa
    # abcabccbacbcabcabcabcabacbcbacbcabcabcbaabca aabbccddaabbccddaabbccddaabbccddaabbccddaabbccddaa

    words = "tt"
    extra = "a"
    setup1 = "from __main__ import palindrome"
    code1 = "palindrome(\"abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz\")"
    time = timeit.timeit(setup=setup1, stmt=code1, number=1000)
    print(time)

    # for x in range(22):
    #     words = extra + words + extra
    #     setup1 = "from __main__ import palindrome2"
    #     code1 = "palindrome2(\"{0}\")".format(words)
    #     print(code1)
    #     time = timeit.timeit(setup=setup1, stmt=code1, number=1)
    #     print(time)
