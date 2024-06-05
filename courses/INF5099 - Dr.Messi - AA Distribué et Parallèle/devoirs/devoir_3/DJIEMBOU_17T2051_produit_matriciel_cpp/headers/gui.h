#ifndef GUI_H
#define GUI_H


/** \methods is_integer
 * \brief Type checking
 * 
 * This check wheter an input chaine can be cast to int
 */
bool is_integer(const char *chaine);

/** \methods get_integer
 * \brief IHM
 * 
 * This get an integer input from user
 */
int get_integer();

/** \methods gui
 * \brief IHM
 * 
 * This run all the main program logic
 */
void gui();

#endif