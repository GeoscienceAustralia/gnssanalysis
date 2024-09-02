import unittest

from gnssanalysis.solution_types import SolutionType, SolutionTypes


class TestSolutionType(unittest.TestCase):
    def test_shortname_to_solution_type(self):
        self.assertEqual(SolutionTypes.from_name("ULT"), SolutionTypes.ULT)
        self.assertEqual(SolutionTypes.from_name("RAP"), SolutionTypes.RAP)
        self.assertEqual(SolutionTypes.from_name("UNK"), SolutionTypes.UNK)
        # AssertRaises can be used either as a context manager, or by breaking out the function arguments.
        # Note we're not *calling* the function, we're passing the function *so it can be called* by the handler.
        self.assertRaises(ValueError, SolutionTypes.from_name, name="noo")
        self.assertRaises(ValueError, SolutionTypes.from_name, name="rapid")
        self.assertRaises(ValueError, SolutionTypes.from_name, name="")
        self.assertRaises(ValueError, SolutionTypes.from_name, name=" ")

    def test_immutability(self):
        def update_base_attribute():
            SolutionType.name = "someNewValue"

        # Note that *contents* of a list can still be modified, despite the more general
        # protections provided by the metaclass

        def update_enum_attribute_new():
            SolutionTypes.ULT.name = "someBogusValue"

        self.assertRaises(AttributeError, update_base_attribute)
        self.assertRaises(AttributeError, update_enum_attribute_new)

        def instantiate_solution_generic():
            SolutionType()

        def instantiate_solution_specific():
            SolutionTypes.RAP()

        def instantiate_solution_helper():
            SolutionTypes()

        self.assertRaises(Exception, instantiate_solution_generic)
        self.assertRaises(Exception, instantiate_solution_specific)
        self.assertRaises(Exception, instantiate_solution_helper)

    def test_equality(self):
        self.assertEqual(SolutionTypes.RAP, SolutionTypes.RAP, "References to same solution type class should be equal")
        self.assertEqual(
            SolutionTypes.from_name("RAP"),
            SolutionTypes.from_name("rap"),
            "from_name should give equal results each time, also regardless of case of input",
        )
        self.assertNotEqual(SolutionTypes.RAP, SolutionTypes.UNK, "Non-matching solution types should be unequal")
