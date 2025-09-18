#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

int isCoreKeyword(const char *str) {
    const char *keywords[] = {
        "auto", "break", "case", "char", "const", "continue", "default", "do",
        "double", "else", "enum", "extern", "float", "for", "goto", "if",
        "int", "long", "register", "return", "short", "signed", "sizeof", "static",
        "struct", "switch", "typedef", "union", "unsigned", "void", "volatile", "while",
        "main", "printf", "scanf"
    };
    int num_keywords = sizeof(keywords) / sizeof(keywords[0]);
    for (int i = 0; i < num_keywords; i++) {
        if (strcmp(str, keywords[i]) == 0) {
            return 1;
        }
    }
    return 0;
}

int isPreprocessorKeyword(const char *str) {
    if (strcmp(str, "include") == 0 || strcmp(str, "define") == 0 || strcmp(str, "ifdef") == 0) {
        return 1;
    }
    return 0;
}

void print_token(const char* token, const char* type) {
    printf("  Token: %-20s Type: %s\n", token, type);
}

void lexical_analyzer(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    char line[512];
    int line_num = 0;
    
    printf("--------------------------------------------------\n");
    printf("Lexical Analysis for file: %s\n", filename);
    printf("--------------------------------------------------\n");

    while (fgets(line, sizeof(line), file)) {
        line_num++;
        printf("\n--- Line %d: %s", line_num, line);
        
        int i = 0;
        while (line[i] != '\0') {
            if (isspace(line[i])) {
                i++;
                continue;
            }

            if (line[i] == '#') {
                print_token("#", "SYMBOL");
                i++;
                continue;
            }

            if (line[i] == '/' && line[i+1] == '/') {
                break; 
            }

            if (line[i] == '"') {
                print_token("\"", "SYMBOL"); i++;
                char buffer[256]; int j = 0;
                while (line[i] != '"' && line[i] != '\0') {
                    if (line[i] == '\\' && line[i+1] == '"') { buffer[j++] = line[i++]; buffer[j++] = line[i++]; } 
                    else { buffer[j++] = line[i++]; }
                }
                buffer[j] = '\0';
                if (strlen(buffer) > 0) { print_token(buffer, "STRING_LITERAL"); }
                if (line[i] == '"') { print_token("\"", "SYMBOL"); i++; } 
                else { printf("  Error: Unterminated string literal on line %d\n", line_num); }
                continue;
            }
            if (line[i] == '\'') {
                 print_token("'", "SYMBOL"); i++;
                char buffer[5]; int j = 0;
                while (line[i] != '\'' && line[i] != '\0') { buffer[j++] = line[i++]; }
                buffer[j] = '\0';
                print_token(buffer, "CHAR_LITERAL");
                if (line[i] == '\'') { print_token("'", "SYMBOL"); i++; } 
                else { printf("  Error: Unterminated char literal on line %d\n", line_num); }
                continue;
            }

            char op_buffer[3] = {line[i], line[i+1], '\0'};
            if (strcmp(op_buffer, "==") == 0 || strcmp(op_buffer, "!=") == 0 ||
                strcmp(op_buffer, ">=") == 0 || strcmp(op_buffer, "<=") == 0 ||
                strcmp(op_buffer, "&&") == 0 || strcmp(op_buffer, "||") == 0) {
                print_token(op_buffer, "OPERATOR");
                i += 2;
                continue;
            }

            if (isalpha(line[i]) || line[i] == '_') {
                char buffer[100];
                int j = 0;
                while (isalnum(line[i]) || line[i] == '_') {
                    buffer[j++] = line[i++];
                }
                buffer[j] = '\0';

                if (isPreprocessorKeyword(buffer)) {
                    print_token(buffer, "KEYWORD");
                } else if (isCoreKeyword(buffer)) {
                    print_token(buffer, "KEYWORD");
                } else {
                    print_token(buffer, "IDENTIFIER");
                }
                continue;
            } 
            else if (line[i] == '<') {
                char buffer[100];
                int j = 0;
                while (line[i] != '>' && line[i] != '\0' && line[i] != '\n') {
                    buffer[j++] = line[i++];
                }
                if (line[i] == '>') {
                    buffer[j++] = line[i++];
                }
                buffer[j] = '\0';
                print_token(buffer, "HEADER_FILE");
                continue;
            }
            else if (isdigit(line[i]) || (line[i] == '.' && isdigit(line[i+1]))) {
                char buffer[100]; int j = 0;
                while (isdigit(line[i]) || line[i] == '.') { buffer[j++] = line[i++]; }
                buffer[j] = '\0';
                print_token(buffer, "CONSTANT");
                continue;
            } else {
                char ch_str[2] = {line[i], '\0'};
                const char* operators = "+-*/%=!&|";
                const char* symbols = "(){}[] ,;"; 
                if (strchr(operators, line[i]) != NULL) {
                    print_token(ch_str, "OPERATOR");
                } else if (strchr(symbols, line[i]) != NULL) {
                    print_token(ch_str, "SYMBOL");
                }
                i++;
            }
        }
    }
    fclose(file);
}

int main() {
    char filename[] = "input.c";
    lexical_analyzer(filename);
    return 0;
}
