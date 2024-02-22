"""Provides routines for manipulating found constraints."""
from typing import Dict
import copy
from .hints import Constraints


def reduce_constraint_sets(constraints: Constraints) -> Constraints:
    r"""Reduces constraints to disjoint constraints.

    Reduces a set of frozensets of constrained sites into a set of larger disjoint
    frozensets of constrained sites.

    If a single atom has a constrained bonds to two separate atoms, the list of
    bond constraints does not make it clear that all three of these atoms are
    (for the purpose of force mappings in this module ) all constrained
    together. This method replaces the two bond constraint entries with a 3 atom
    constraint entry, but does so for all atoms and all sized constrains such
    that the returned list of constraint's members are all disjoint.

    Arguments:
    ---------
    constraints (set of frozensets of integers):
        Each member set contains indices of atoms which are constrained relative
        to each other

    Returns:
    -------
    set of frozensets of integers.

    Example:
    -------
        {{1,2},{2,3},{4,5}}
        is transformed into
        {{1,2,3},{4,5}}

        In other words, {1,2} and {2,3} were combined because they both
        contained 2, while {4,5} was untouched as it did not share any elements
        with any other sets.

    NOTE: This function has complicated flow and is not proven to be correct. It
    seems to be a form of flood search using breadth first search should be
    revised.
    """
    constraints_copy = copy.copy(constraints)
    agged_constraints = set()

    if len(constraints) <= 1:
        return constraints_copy

    # this control flow is very bad, but we do not a good refactor yet.

    # We pop an element from constraints copy, and see if it has a non-empty
    # intersection with any other elements in constraints copy.  If so, we union
    # those sets into our selected element. The elements we had similarity with
    # in constraints copy are removed.  We repeat this process until we do not
    # see any candidates to add, at which point we add our selected (aggregated)
    # element to a new set, take the next element from the constraints copy, and
    # begin again. The new set is returned when we run out of elements in
    # constraints copy to process.

    new = frozenset(constraints_copy.pop())
    second_try = False
    while True:
        to_add = [x for x in constraints_copy if new.intersection(x)]
        new = new.union(*to_add)
        constraints_copy.difference_update(to_add)
        if not to_add:
            agged_constraints.add(new)
            if second_try:
                second_try = False
                try:
                    new = frozenset(constraints_copy.pop())
                except KeyError:
                    break
            else:
                second_try = True
    return agged_constraints


def constraint_lookup_dict(constraints: Constraints) -> Dict[int, int]:
    r"""Transform constraints to a dictionary connecting each member to a parent.

    Transforms a set of frozensets of constraints to a dictionary connecting
    each set member to a master member.

    The smallest member of each member set is designated as the parent member.
    Each remaining element in that set is added to the return dictionary as a
    key which points to its parent member.

    Arguments:
    ---------
    constraints:
        Constraints to collapse

    Example:
    -------
    constraints = {{1,2,3},{4,5},{6,7}}
    is transformed into the following dictionary:
    {
        3:1
        2:1
        5:4
        7:6
    }
    In other words, 1 is the parent member of the first member set, so 2 and
    3 point to 1, etc.

    This is useful when setting up matrices for the quadratic programming
    problem when molecular constraint are present.
    """
    mapping = {}
    for group in constraints:
        sites = sorted(group)
        anchor = sites[0]
        _ = [mapping.update({s: anchor}) for s in sites[1:]]
    return mapping
